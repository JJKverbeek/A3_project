"""Ask Claude Haiku each question, extract a self-confidence score,
then judge the answer two ways (blind and with the gold reference).
Repeats each question N_RUNS times for reliability analysis.

Output: results.csv with one row per (question, run).
"""

import os
import re
import time
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5-20251001"
N_RUNS = 3
SLEEP = 0.3  # seconds between calls; gentle on rate limits

load_dotenv()
client = Anthropic()


def ask(prompt: str, temperature: float = 0.7) -> str:
    """One-shot call. Each call is a fresh conversation (no history)."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def parse_confidence(text: str) -> int | None:
    m = re.search(r"Confidence:\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_verdict(text: str) -> str:
    """Return 'CORRECT' or 'INCORRECT' based on the first such word."""
    upper = text.upper()
    i_c = upper.find("CORRECT")
    i_i = upper.find("INCORRECT")
    if i_i != -1 and (i_c == -1 or i_i < i_c):
        return "INCORRECT"
    if i_c != -1:
        return "CORRECT"
    return "UNKNOWN"


def get_answer_and_confidence(question: str) -> tuple[str, int | None, str]:
    raw = ask(
        f"Answer the following question briefly (1-2 sentences). "
        f"Then on a new line write only 'Confidence: X/100' where X is your "
        f"confidence in the correctness of your answer.\n\n"
        f"Question: {question}"
    )
    conf = parse_confidence(raw)
    answer = re.split(r"Confidence:", raw, flags=re.IGNORECASE)[0].strip()
    return answer, conf, raw


def blind_judge(question: str, answer: str) -> str:
    raw = ask(
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        f"Is this answer factually correct? "
        f"Reply with CORRECT or INCORRECT followed by a one-sentence reason.",
        temperature=0.0,
    )
    return parse_verdict(raw)


def gold_judge(question: str, answer: str, gold: str) -> str:
    raw = ask(
        f"Question: {question}\n"
        f"Reference answer: {gold}\n"
        f"Proposed answer: {answer}\n\n"
        f"Is the proposed answer consistent with the reference? "
        f"Reply CORRECT or INCORRECT followed by a one-sentence reason.",
        temperature=0.0,
    )
    return parse_verdict(raw)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")

    questions = pd.read_csv("questions.csv")
    rows = []
    total = len(questions) * N_RUNS

    for run in range(N_RUNS):
        for _, q in questions.iterrows():
            i = len(rows) + 1
            print(f"[{i}/{total}] q_id={q.q_id} run={run}")
            answer, conf, _ = get_answer_and_confidence(q.question)
            time.sleep(SLEEP)
            blind = blind_judge(q.question, answer)
            time.sleep(SLEEP)
            gold = gold_judge(q.question, answer, q.gold)
            time.sleep(SLEEP)
            rows.append({
                "q_id": q.q_id,
                "run": run,
                "question": q.question,
                "gold": q.gold,
                "answer": answer,
                "confidence": conf,
                "blind_judge": blind,
                "gold_judge": gold,
            })
            # incremental save in case of interruption
            pd.DataFrame(rows).to_csv("results.csv", index=False)

    print(f"\nDone. {len(rows)} rows written to results.csv")


if __name__ == "__main__":
    main()
