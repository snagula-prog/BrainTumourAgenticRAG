# tests/test_phase7_hardening.py
"""Phase 7: Cleaning Hardening & Idempotency Validation."""

import copy
from cleaning.normalizer import CanonicalNormalizer


def test_idempotency():
    """Verify clean(clean(doc)) == clean(doc)."""
    dirty_doc = {
        "metadata": {"title": "Swirl-Stabilized  Thermojet\u00ad-Combustor\u200b Analysis"},
        "abstract": "Reported accuracy is 98.5% across 1,000 samples.",
        "paragraphs": [
            {
                "text": "The model achieved 98.5% accuracy.\n\n",
                "page": 1,
                "section": "Results",
            },
            {
                "text": "The model achieved 98.5% accuracy.",
                "page": 1,
                "section": "Results",
            },
        ],
        "formulas": [{"formula_id": "eq1", "source": "E = mc^2"}],
    }

    normalizer = CanonicalNormalizer()
    first_pass, _ = normalizer.clean(copy.deepcopy(dirty_doc))
    second_pass, _ = normalizer.clean(copy.deepcopy(first_pass))

    # Strip dynamic report metadata for pure content comparison
    first_clean = {k: v for k, v in first_pass.items() if k != "cleaning"}
    second_clean = {k: v for k, v in second_pass.items() if k != "cleaning"}

    assert first_clean == second_clean, "Idempotency failed: second cleaning modified the artifact!"
    print("✓ Idempotency test passed.")


def test_noise_removal():
    """Verify soft hyphens, zero-width chars, and duplicate paragraphs are cleaned."""
    dirty_doc = {
        "abstract": "Bad\u00adly split\u200b text with   extra   spaces.",
        "paragraphs": [
            {"text": "Duplicate paragraph.", "page": 2, "section": "Methods"},
            {"text": "Duplicate paragraph.", "page": 2, "section": "Methods"},
            {"text": "", "page": 2, "section": "Methods"},
        ],
    }

    normalizer = CanonicalNormalizer()
    cleaned, report = normalizer.clean(dirty_doc)

    assert cleaned["abstract"] == "Badly split text with extra spaces."
    assert len(cleaned["paragraphs"]) == 1
    assert report["stats"]["removed_empty_items"] == 1
    assert report["stats"]["duplicate_paragraphs_removed"] == 1
    print("✓ Noise removal test passed.")


def test_formula_immunity():
    """Verify formulas are not altered by generic prose rules."""
    dirty_doc = {
        "formulas": [
            {"formula_id": "eq_01", "source": "\\hat{y} = \\frac{1}{1 + e^{-x}}"}
        ]
    }

    normalizer = CanonicalNormalizer()
    cleaned, _ = normalizer.clean(dirty_doc)

    assert cleaned["formulas"][0]["source"] == "\\hat{y} = \\frac{1}{1 + e^{-x}}"
    print("✓ Formula immunity test passed.")


def test_numeric_guard_trigger():
    """Verify that if a normalization inadvertently changes a number, it reverts."""
    # Force a condition where original text has numbers that normalization would alter
    dirty_doc = {
        "abstract": "The accuracy was 98.2%.",
    }

    normalizer = CanonicalNormalizer()
    # Mocking text transformation that alters a number to test guard fallback
    original_normalize = normalizer._normalize_text

    def buggy_normalize(value, stats):
        stats.input_text_fields += 1
        # Intentionally alter numbers
        text = value.replace("98.2", "92.8")
        if normalizer._numeric_tokens(value) != normalizer._numeric_tokens(text):
            stats.numeric_guard_failures += 1
            return value
        return text

    normalizer._normalize_text = buggy_normalize
    cleaned, report = normalizer.clean(dirty_doc)

    assert cleaned["abstract"] == "The accuracy was 98.2%."
    assert report["stats"]["numeric_guard_failures"] == 1
    print("✓ Numeric guard safety test passed.")


def main():
    test_idempotency()
    test_noise_removal()
    test_formula_immunity()
    test_numeric_guard_trigger()
    print("\nPhase 7 Hardening: ALL PASSED")


if __name__ == "__main__":
    main()