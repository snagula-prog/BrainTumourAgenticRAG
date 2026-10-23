"""Deterministic validation for template-generated plans."""
from __future__ import annotations

from .catalog import CONTRACTS, EXTERNAL_INPUTS, NUMERIC_OPERATIONS, TERMINAL_OPERATIONS
from .schemas import Plan


class PlanValidationError(ValueError):
    pass


class PlanValidator:
    def validate(self, plan: Plan) -> Plan:
        if len(plan.steps) > plan.budget.max_steps:
            raise PlanValidationError("Plan exceeds its configured step budget.")

        step_ids: set[str] = set()
        available = set(EXTERNAL_INPUTS)
        numeric_since_validation = False
        last_numeric_index = -1
        last_numeric_validation_index = -1

        for index, step in enumerate(plan.steps):
            if step.step_id in step_ids:
                raise PlanValidationError(f"Duplicate step_id: {step.step_id}")
            step_ids.add(step.step_id)

            expected_dependencies = [] if index == 0 else [plan.steps[index - 1].step_id]
            if step.depends_on != expected_dependencies:
                raise PlanValidationError(
                    f"Step {step.step_id} must depend on the previous template step: {expected_dependencies}"
                )

            contract = CONTRACTS.get(step.operation)
            if contract is None:
                raise PlanValidationError(f"Operation is not registered: {step.operation}")
            if set(step.requires) != set(contract.requires):
                raise PlanValidationError(f"Incorrect requires contract for {step.step_id} ({step.operation}).")
            if set(step.requires_any) != set(contract.requires_any):
                raise PlanValidationError(f"Incorrect requires_any contract for {step.step_id} ({step.operation}).")
            if set(step.produces) != set(contract.produces):
                raise PlanValidationError(f"Incorrect produces contract for {step.step_id} ({step.operation}).")

            missing = set(step.requires) - available
            if missing:
                raise PlanValidationError(f"Step {step.step_id} requires unavailable artifacts: {sorted(missing)}")
            if step.requires_any and not (set(step.requires_any) & available):
                raise PlanValidationError(
                    f"Step {step.step_id} requires at least one of: {sorted(step.requires_any)}"
                )

            if step.operation in NUMERIC_OPERATIONS:
                numeric_since_validation = True
                last_numeric_index = index
            if step.operation == "validate_derived_values":
                if not numeric_since_validation:
                    raise PlanValidationError("validate_derived_values has no preceding numeric operation.")
                numeric_since_validation = False
                last_numeric_validation_index = index

            available.update(step.produces)

        if not plan.steps or plan.steps[-1].operation not in TERMINAL_OPERATIONS:
            raise PlanValidationError("Plan must end with a registered terminal operation.")
        if numeric_since_validation or (last_numeric_index > last_numeric_validation_index):
            raise PlanValidationError("Numeric outputs must be followed by validate_derived_values before termination.")

        if plan.template_id == "T_CLARIFY" and [s.operation for s in plan.steps] != ["ask_clarification"]:
            raise PlanValidationError("T_CLARIFY must contain only ask_clarification.")
        if plan.template_id == "T_DECLINE" and [s.operation for s in plan.steps] != ["decline"]:
            raise PlanValidationError("T_DECLINE must contain only decline.")
        if plan.template_id == "T_META" and [s.operation for s in plan.steps] != ["search_metadata", "render_metadata_response"]:
            raise PlanValidationError("T_META must search metadata and render its response.")
        if plan.template_id not in {"T_CLARIFY", "T_DECLINE", "T_META"} and plan.steps[-1].operation != "verify":
            raise PlanValidationError("Research-answer templates must end with verify.")
        if "render_chart" in [s.operation for s in plan.steps]:
            operations = [s.operation for s in plan.steps]
            if not (operations.index("validate_chart_data") < operations.index("build_chart_spec") < operations.index("render_chart")):
                raise PlanValidationError("Charts require validated data and a validated ChartSpec before rendering.")
        return plan
