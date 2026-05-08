"""Download a 50-question subset of TruthfulQA and save to questions.csv."""

import pandas as pd
from datasets import load_dataset

N_QUESTIONS = 50
OUTPUT = "questions.csv"

ds = load_dataset("truthful_qa", "generation", split="validation")
subset = ds.select(range(N_QUESTIONS))

pd.DataFrame({
    "q_id": list(range(N_QUESTIONS)),
    "question": subset["question"],
    "gold": subset["best_answer"],
}).to_csv(OUTPUT, index=False)

print(f"Wrote {N_QUESTIONS} questions to {OUTPUT}")
