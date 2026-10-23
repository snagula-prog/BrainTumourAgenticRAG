"""Budgeted executor for validated, allowlisted plans.

The executor owns sequencing, runtime artifact checks, budgets, output-contract
checks, and step records. It does not implement or replace retrieval itself:
project-specific adapters are injected as operation handlers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from query_analysis.schemas import QueryAnalysis
from planner import Plan, PlanValidator
from planner.catalog import EXTERNAL_INPUTS
from planner.schemas import PlanStep

from .tool_contracts import ResourceKind, TOOL_CONTRACTS


class OperationHandler(Protocol):
    """An adapter receives read-only context and returns declared artifacts only."""

    def __call__(self, context: "StepExecutionContext") -> Mapping[str, Any]:
        """Return a mapping whose keys exactly match the operation's `produces`."""
        ...


@dataclass(frozen=True)
class StepExecutionContext:
    plan: Plan
    step: PlanStep
    query: str
    analysis: QueryAnalysis
    scope: Any
    artifacts: Mapping[str, Any]
    elapsed_seconds: float
    remaining_seconds: float
    retrieval_calls_used: int
    llm_calls_used: int

    def has_artifact(self, name: str) -> bool:
        """Return whether an artifact exists, regardless of whether its value is empty."""
        return name in self.artifacts

    def get_artifact(self, name: str, default: Any = None) -> Any:
        return self.artifacts.get(name, default)


@dataclass(frozen=True)
class StepExecutionRecord:
    step_id: str
    operation: str
    status: str
    elapsed_seconds: float
    output_keys: tuple[str, ...] = ()
    error: str | None = None


@dataclass
class ExecutionResult:
    plan_id: str
    status: str
    artifacts: dict[str, Any]
    step_records: list[StepExecutionRecord]
    elapsed_seconds: float
    retrieval_calls_used: int = 0
    llm_calls_used: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


class ExecutorConfigurationError(RuntimeError):
    """An operation has no handler or the executor was configured incorrectly."""


class Executor:
    """Execute plans in dependency order with fail-closed validation and budgets.

    Args:
        handlers: Mapping from allowlisted operation names to callable adapters.
        validator: Planner validator. The plan is revalidated immediately before use.
        resource_kinds: Optional per-operation overrides for budget accounting.

    A retrieval/LLM budget unit means one logical registered handler invocation,
    not necessarily one HTTP call hidden inside a composite adapter.
    """

    def __init__(
        self,
        handlers: Mapping[str, OperationHandler],
        *,
        validator: PlanValidator | None = None,
        resource_kinds: Mapping[str, ResourceKind] | None = None,
    ) -> None:
        unknown = set(handlers) - set(TOOL_CONTRACTS)
        if unknown:
            raise ExecutorConfigurationError(
                f"Handlers registered for unknown operations: {sorted(unknown)}"
            )
        non_callable = [name for name, handler in handlers.items() if not callable(handler)]
        if non_callable:
            raise ExecutorConfigurationError(
                f"Handlers must be callable: {sorted(non_callable)}"
            )
        self.handlers = dict(handlers)
        self.validator = validator or PlanValidator()
        self.resource_kinds: dict[str, ResourceKind] = {
            name: contract.resource_kind for name, contract in TOOL_CONTRACTS.items()
        }
        if resource_kinds:
            invalid_kinds = {
                name: kind for name, kind in resource_kinds.items()
                if name not in TOOL_CONTRACTS or kind not in {"none", "retrieval", "llm"}
            }
            if invalid_kinds:
                raise ExecutorConfigurationError(
                    f"Invalid resource-kind overrides: {invalid_kinds}"
                )
            self.resource_kinds.update(resource_kinds)

    def execute(
        self,
        plan: Plan,
        *,
        query: str,
        analysis: QueryAnalysis,
        initial_artifacts: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Run one validated plan; stop immediately at the first failure.

        `query`, `analysis`, and `scope` are protected external inputs and cannot be
        overwritten via `initial_artifacts`. Runtime requirements are tested using
        key presence, not truthiness, so empty lists such as `facts=[]` are valid
        artifacts and can be gap-filled.
        """
        start = perf_counter()
        self.validator.validate(plan)
        self._validate_scope_match(plan, analysis)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        supplied = dict(initial_artifacts or {})
        protected = EXTERNAL_INPUTS.intersection(supplied)
        if protected:
            raise ValueError(
                "initial_artifacts cannot override protected inputs: "
                + ", ".join(sorted(protected))
            )

        # Preflight all handlers before running any step to avoid partial execution
        # caused solely by incomplete dependency injection.
        missing_handlers = sorted({step.operation for step in plan.steps} - set(self.handlers))
        if missing_handlers:
            message = f"No handler registered for planned operation(s): {missing_handlers}"
            return ExecutionResult(
                plan_id=plan.plan_id,
                status="failed",
                artifacts=supplied,
                step_records=[],
                elapsed_seconds=perf_counter() - start,
                error=message,
            )

        artifacts: dict[str, Any] = {
            **supplied,
            "query": query.strip(),
            "analysis": analysis,
            "scope": analysis.scope,
        }
        records: list[StepExecutionRecord] = []
        completed_ids: set[str] = set()
        retrieval_calls = 0
        llm_calls = 0
        wall_budget = float(plan.budget.max_wall_time_seconds)

        for step in plan.steps:
            step_started = perf_counter()
            elapsed_before = step_started - start
            if elapsed_before >= wall_budget:
                record = StepExecutionRecord(
                    step_id=step.step_id,
                    operation=step.operation,
                    status="failed",
                    elapsed_seconds=0.0,
                    error="Wall-time budget exhausted before step execution.",
                )
                records.append(record)
                return self._result(plan, "failed", artifacts, records, start,
                                    retrieval_calls, llm_calls, record.error)

            # Recheck dependency completion and artifact presence at runtime; this
            # guards against handler failures and inconsistent intermediate state.
            missing_dependencies = set(step.depends_on) - completed_ids
            if missing_dependencies:
                error = f"Uncompleted dependencies: {sorted(missing_dependencies)}"
                return self._failed_step_result(plan, step, error, step_started,
                    artifacts, records, start, retrieval_calls, llm_calls)

            contract = TOOL_CONTRACTS[step.operation]
            missing_required = contract.requires - artifacts.keys()
            if missing_required:
                error = f"Missing required artifact(s): {sorted(missing_required)}"
                return self._failed_step_result(plan, step, error, step_started,
                    artifacts, records, start, retrieval_calls, llm_calls)
            if contract.requires_any and not (contract.requires_any & artifacts.keys()):
                error = f"Requires at least one artifact from: {sorted(contract.requires_any)}"
                return self._failed_step_result(plan, step, error, step_started,
                    artifacts, records, start, retrieval_calls, llm_calls)

            resource_kind = self.resource_kinds[step.operation]
            if resource_kind == "retrieval":
                if retrieval_calls >= plan.budget.max_retrieval_calls:
                    error = "Retrieval-call budget exhausted."
                    return self._failed_step_result(plan, step, error, step_started,
                        artifacts, records, start, retrieval_calls, llm_calls)
                retrieval_calls += 1
            elif resource_kind == "llm":
                if llm_calls >= plan.budget.max_llm_calls:
                    error = "LLM-call budget exhausted."
                    return self._failed_step_result(plan, step, error, step_started,
                        artifacts, records, start, retrieval_calls, llm_calls)
                llm_calls += 1

            context = StepExecutionContext(
                plan=plan,
                step=step,
                query=query.strip(),
                analysis=analysis,
                scope=analysis.scope,
                artifacts=MappingProxyType(dict(artifacts)),
                elapsed_seconds=elapsed_before,
                remaining_seconds=max(0.0, wall_budget - elapsed_before),
                retrieval_calls_used=retrieval_calls,
                llm_calls_used=llm_calls,
            )
            try:
                output = self.handlers[step.operation](context)
                if not isinstance(output, Mapping):
                    raise TypeError("Handler output must be a mapping of artifact names to values.")
                expected = contract.produces
                actual = set(output.keys())
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    raise ValueError(
                        f"Output contract mismatch; missing={missing}, unexpected={extra}."
                    )
                step_elapsed = perf_counter() - step_started
                if perf_counter() - start > wall_budget:
                    raise TimeoutError(
                        "Step finished after the wall-time budget; output was not committed. "
                        "Handlers should use context.remaining_seconds for their own timeouts."
                    )
            except Exception as exc:
                record = StepExecutionRecord(
                    step_id=step.step_id,
                    operation=step.operation,
                    status="failed",
                    elapsed_seconds=perf_counter() - step_started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                records.append(record)
                return self._result(plan, "failed", artifacts, records, start,
                                    retrieval_calls, llm_calls, record.error)

            # Commit exactly the artifacts declared by the tool contract.
            artifacts.update(dict(output))
            records.append(StepExecutionRecord(
                step_id=step.step_id,
                operation=step.operation,
                status="completed",
                elapsed_seconds=step_elapsed,
                output_keys=tuple(sorted(expected)),
            ))
            completed_ids.add(step.step_id)

        return self._result(plan, "completed", artifacts, records, start,
                            retrieval_calls, llm_calls, None)

    @staticmethod
    def _validate_scope_match(plan: Plan, analysis: QueryAnalysis) -> None:
        scope = analysis.scope
        if plan.scope_mode != scope.mode:
            raise ValueError("Plan scope_mode does not match validated QueryAnalysis scope.")
        if plan.paper_ids != list(scope.paper_ids):
            raise ValueError("Plan paper_ids do not match validated QueryAnalysis scope.")
        if plan.filters != dict(scope.filters):
            raise ValueError("Plan filters do not match validated QueryAnalysis scope.")

    @staticmethod
    def _failed_step_result(
        plan: Plan,
        step: PlanStep,
        error: str,
        step_started: float,
        artifacts: dict[str, Any],
        records: list[StepExecutionRecord],
        start: float,
        retrieval_calls: int,
        llm_calls: int,
    ) -> ExecutionResult:
        record = StepExecutionRecord(
            step_id=step.step_id,
            operation=step.operation,
            status="failed",
            elapsed_seconds=perf_counter() - step_started,
            error=error,
        )
        records.append(record)
        return Executor._result(plan, "failed", artifacts, records, start,
                                retrieval_calls, llm_calls, error)

    @staticmethod
    def _result(
        plan: Plan,
        status: str,
        artifacts: dict[str, Any],
        records: list[StepExecutionRecord],
        start: float,
        retrieval_calls: int,
        llm_calls: int,
        error: str | None,
    ) -> ExecutionResult:
        return ExecutionResult(
            plan_id=plan.plan_id,
            status=status,
            artifacts=dict(artifacts),
            step_records=list(records),
            elapsed_seconds=perf_counter() - start,
            retrieval_calls_used=retrieval_calls,
            llm_calls_used=llm_calls,
            error=error,
        )
