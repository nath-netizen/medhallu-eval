import pytest
from medhallu.medhallu import _build_input, _record_to_samples

SAMPLE_RECORD = {
    "Knowledge": (
        "Metformin is a biguanide antihyperglycemic agent used as first-line "
        "pharmacotherapy for type 2 diabetes mellitus."
    ),
    "Question": "What pharmacological class does metformin belong to?",
    "Ground Truth": "Metformin belongs to the biguanide class of antihyperglycemic agents.",
    "Hallucinated Answer": "Metformin is a sulfonylurea that stimulates pancreatic insulin secretion.",
    "Difficulty": "easy",
}


# --- _build_input ---

def test_build_input_contains_knowledge():
    result = _build_input(
        SAMPLE_RECORD["Knowledge"],
        SAMPLE_RECORD["Question"],
        SAMPLE_RECORD["Ground Truth"],
    )
    assert "biguanide antihyperglycemic" in result


def test_build_input_contains_question():
    result = _build_input(
        SAMPLE_RECORD["Knowledge"],
        SAMPLE_RECORD["Question"],
        SAMPLE_RECORD["Ground Truth"],
    )
    assert "What pharmacological class" in result


def test_build_input_contains_answer():
    result = _build_input(
        SAMPLE_RECORD["Knowledge"],
        SAMPLE_RECORD["Question"],
        SAMPLE_RECORD["Ground Truth"],
    )
    assert "Metformin belongs to the biguanide class" in result


def test_build_input_hallucinated_answer_differs():
    factual = _build_input(
        SAMPLE_RECORD["Knowledge"],
        SAMPLE_RECORD["Question"],
        SAMPLE_RECORD["Ground Truth"],
    )
    hallucinated = _build_input(
        SAMPLE_RECORD["Knowledge"],
        SAMPLE_RECORD["Question"],
        SAMPLE_RECORD["Hallucinated Answer"],
    )
    assert factual != hallucinated
    assert "sulfonylurea" in hallucinated
    assert "sulfonylurea" not in factual


# --- _record_to_samples ---

def test_record_to_samples_returns_two():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert len(samples) == 2


def test_record_to_samples_ids():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=42)
    assert samples[0].id == "42_factual"
    assert samples[1].id == "42_hallucinated"


def test_record_to_samples_targets():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert samples[0].target == "A"  # factual → A
    assert samples[1].target == "B"  # hallucinated → B


def test_record_to_samples_choices():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    for s in samples:
        assert s.choices == ["factual", "hallucinated"]


def test_record_to_samples_metadata_answer_type():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert samples[0].metadata["answer_type"] == "factual"
    assert samples[1].metadata["answer_type"] == "hallucinated"


def test_record_to_samples_metadata_difficulty():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert samples[0].metadata["difficulty"] == "easy"
    assert samples[1].metadata["difficulty"] == "easy"


def test_record_to_samples_factual_input_contains_ground_truth():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert SAMPLE_RECORD["Ground Truth"] in samples[0].input


def test_record_to_samples_hallucinated_input_contains_hallucinated_answer():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert SAMPLE_RECORD["Hallucinated Answer"] in samples[1].input


def test_record_to_samples_inputs_differ():
    samples = _record_to_samples(SAMPLE_RECORD, base_id=0)
    assert samples[0].input != samples[1].input


def test_record_to_samples_difficulty_capital_D_fallback():
    record = {**SAMPLE_RECORD, "Difficulty": "hard"}
    samples = _record_to_samples(record, base_id=0)
    assert samples[0].metadata["difficulty"] == "hard"


def test_record_to_samples_difficulty_lowercase_fallback():
    record = {k: v for k, v in SAMPLE_RECORD.items() if k != "Difficulty"}
    record["difficulty"] = "medium"
    samples = _record_to_samples(record, base_id=0)
    assert samples[0].metadata["difficulty"] == "medium"


def test_record_to_samples_difficulty_unknown_fallback():
    record = {k: v for k, v in SAMPLE_RECORD.items() if k != "Difficulty"}
    samples = _record_to_samples(record, base_id=0)
    assert samples[0].metadata["difficulty"] == "unknown"


def test_record_to_samples_base_id_increments():
    samples_0 = _record_to_samples(SAMPLE_RECORD, base_id=0)
    samples_7 = _record_to_samples(SAMPLE_RECORD, base_id=7)
    assert samples_0[0].id == "0_factual"
    assert samples_7[0].id == "7_factual"
