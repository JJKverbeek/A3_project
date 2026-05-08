# A3: LLM Self-Confidence + LLM-as-Judge Evaluation

VU NLP 2026, Assignment 3.

## What this does

Tests whether an LLM (Claude Haiku 4.5) can reliably:
1. Score its own confidence (0-100) when answering a factual question.
2. Judge — in a fresh conversation — whether its own answer was correct.

We measure both signals against a gold-judge call (the same model, but given the reference answer) and against multiple reruns to test reproducibility.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and add your real Anthropic API key. The `.env` file is ignored by git so the secret is not committed.
Do not include `.env` in a public repository or submission zip after adding a real key.

## Run

```bash
python prepare_data.py    # downloads TruthfulQA -> questions.csv
python run.py             # ~5-10 min, <$0.20 in API cost -> results.csv
python analyze.py         # prints metrics, writes plots/*.png
```

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Download 50 TruthfulQA questions + gold answers |
| `run.py` | For each question x 3 runs: get answer + confidence, blind-judge it, gold-judge it |
| `analyze.py` | Compute calibration, judge F1, reliability, agreement; save plots |
| `questions.csv` | Generated input |
| `results.csv` | Generated raw outputs |
| `plots/` | Generated calibration, judge-confusion, confidence-vs-judge, and reproducibility plots |
| `writeup.md` | Draft of the 5-section report |

## Dataset

50 questions from TruthfulQA (Lin et al. 2022) — designed to expose factual unreliability in LLMs.
