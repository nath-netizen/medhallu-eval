"""
MedHallu: Benchmark for detecting medical hallucinations in LLM-generated answers

"MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations
in Large Language Models" (2024)
Shrey Pandit, Ashok Vardhan Makkena, Akshay Nambi, et al.
https://arxiv.org/abs/2408.08511

Dataset: UTAustin-AIHealth/MedHallu (pqa_labeled config)
Task: given a PubMed-derived knowledge passage, a question, and a candidate answer,
classify whether the answer is factual or hallucinated.

Each dataset row generates two eval samples — one with the ground-truth answer
(target: factual) and one with the hallucinated answer (target: hallucinated).
Results are reported overall and broken down by difficulty (easy/medium/hard).

Usage:
    inspect eval src/medhallu/medhallu.py@medhallu
    inspect eval src/medhallu/medhallu.py@medhallu --limit 50
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

DATASET_ID = "UTAustin-AIHealth/MedHallu"
DATASET_CONFIG = "pqa_labeled"
# Pin to a specific commit so results are reproducible
DATASET_REVISION = "515060458a945c633debc6fd5baac7764416b724"

EVAL_VERSION = 1

# multiple_choice() substitutes {question} (= Sample.input), {choices}, {letters}
TEMPLATE = (
    "{question}\n\n"
    "{choices}\n\n"
    "The entire content of your response should be of the following format: "
    "'ANSWER: $LETTER' (without quotes) where LETTER is one of {{{letters}}}."
)


@task
def medhallu() -> Task:
    """Inspect Task for the MedHallu medical hallucination detection benchmark."""
    from datasets import load_dataset

    hf_dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        split="test",
        trust_remote_code=False,
    )

    samples: list[Sample] = []
    for i, record in enumerate(hf_dataset):
        samples.extend(_record_to_samples(record, base_id=i))

    return Task(
        dataset=MemoryDataset(samples, name="medhallu"),
        solver=[multiple_choice(template=TEMPLATE)],
        scorer=choice(),
        metadata={"version": EVAL_VERSION},
    )


def _record_to_samples(record: dict[str, Any], base_id: int) -> list[Sample]:
    """Return two Samples per dataset row: one factual, one hallucinated."""
    knowledge = record["Knowledge"]
    question = record["Question"]
    # Dataset uses "Difficulty" (capital D); fall back gracefully
    difficulty = record.get("Difficulty", record.get("difficulty", "unknown"))

    return [
        Sample(
            input=_build_input(knowledge, question, record["Ground Truth"]),
            target="A",
            choices=["factual", "hallucinated"],
            id=f"{base_id}_factual",
            metadata={"difficulty": difficulty, "answer_type": "factual"},
        ),
        Sample(
            input=_build_input(knowledge, question, record["Hallucinated Answer"]),
            target="B",
            choices=["factual", "hallucinated"],
            id=f"{base_id}_hallucinated",
            metadata={"difficulty": difficulty, "answer_type": "hallucinated"},
        ),
    ]


def _build_input(knowledge: str, question: str, answer: str) -> str:
    return (
        "You are a medical expert. Read the knowledge passage and question below, "
        "then evaluate whether the provided answer is factual or hallucinated.\n\n"
        f"Knowledge:\n{knowledge}\n\n"
        f"Question: {question}\n\n"
        f"Answer to evaluate: {answer}"
    )
