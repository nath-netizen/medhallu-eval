# medhallu-eval

Inspect AI implementation of the [MedHallu](https://arxiv.org/abs/2408.08511) benchmark — detecting hallucinations in LLM-generated medical answers.

**Dataset:** `UTAustin-AIHealth/MedHallu` (pqa_labeled)  
**Task:** Classify whether a candidate answer to a PubMed-derived question is factual or hallucinated.  
**Metric:** Accuracy (overall and by difficulty: easy / medium / hard)

## Setup

```bash
pip install -e ".[dev]"
```

## Run

```bash
# Full eval (~2× dataset size — two samples per row)
inspect eval src/medhallu/medhallu.py@medhallu

# Quick smoke test (first 20 rows → 40 samples)
inspect eval src/medhallu/medhallu.py@medhallu --limit 40

# Against a specific model
inspect eval src/medhallu/medhallu.py@medhallu --model openai/gpt-4o
```

## Test

```bash
pytest tests/
```

## Citation

```bibtex
@article{pandit2024medhallu,
  title={MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models},
  author={Pandit, Shrey and Makkena, Ashok Vardhan and Nambi, Akshay and others},
  journal={arXiv preprint arXiv:2408.08511},
  year={2024}
}
```
