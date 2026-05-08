"""Read results.csv, compute metrics, and save analysis plots."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

os.makedirs("plots", exist_ok=True)
df = pd.read_csv("results.csv")

# Treat gold_judge as ground truth.
df["correct"] = (df["gold_judge"] == "CORRECT").astype(int)
df["blind_correct"] = (df["blind_judge"] == "CORRECT").astype(int)


def prf1(pred, true):
    """Treat 'CORRECT' as the positive class."""
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def wilson_interval(successes, n, z=1.96):
    """Approximate 95% Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def annotate_bars(ax, bars, labels, y_offset=0.02):
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ---------- 1. Judge reliability (blind vs gold) ----------
p, r, f1 = prf1(df["blind_correct"], df["correct"])
acc = (df["blind_judge"] == df["gold_judge"]).mean()
print("=== Blind judge vs gold judge ===")
print(f"Accuracy: {acc:.3f}   Precision: {p:.3f}   Recall: {r:.3f}   F1: {f1:.3f}")

# ---------- 2. Calibration ----------
print("\n=== Calibration (confidence -> accuracy) ===")
df_c = df.dropna(subset=["confidence"]).copy()
df_c["bin"] = pd.cut(df_c["confidence"], bins=[0, 20, 40, 60, 80, 100],
                     labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
                     include_lowest=True)
cal = df_c.groupby("bin", observed=True).agg(
    n=("correct", "size"),
    accuracy=("correct", "mean"),
    mean_conf=("confidence", "mean"),
    blind_correct_rate=("blind_correct", "mean"),
)
cal["ci_low"], cal["ci_high"] = zip(
    *[wilson_interval(round(row.accuracy * row.n), row.n) for row in cal.itertuples()]
)
print(cal)

fig, ax = plt.subplots(figsize=(7, 4.5))
x = range(len(cal))
bars = ax.bar(
    x,
    cal["accuracy"],
    color="#4C78A8",
    alpha=0.85,
    label="Actual accuracy",
)
err_low = cal["accuracy"] - cal["ci_low"]
err_high = cal["ci_high"] - cal["accuracy"]
ax.errorbar(
    x,
    cal["accuracy"],
    yerr=[err_low, err_high],
    fmt="none",
    ecolor="#222222",
    capsize=4,
    linewidth=1,
    label="95% CI",
)


def _mid(label):
    lo, hi = label.split("-")
    return (int(lo) + int(hi)) / 200


diag = [_mid(str(b)) for b in cal.index]
ax.plot(x, diag, color="#222222", linestyle="--", label="Perfect calibration")
ax.scatter(
    x,
    cal["mean_conf"] / 100,
    color="#F58518",
    zorder=3,
    label="Mean reported confidence",
)
annotate_bars(ax, bars, [f"n={n}" for n in cal["n"]])
ax.set_xticks(list(x), cal.index.astype(str))
ax.set_xlabel("Self-reported confidence bin")
ax.set_ylabel("Share of answers")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_title(f"Calibration: confidence is informative but inflated (n={len(df_c)})")
ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig("plots/calibration.png", dpi=120)
print("\nSaved plots/calibration.png")

# ---------- 3. Blind-vs-gold judge confusion matrix ----------
confusion = pd.crosstab(
    df["gold_judge"],
    df["blind_judge"],
    rownames=["Gold judge"],
    colnames=["Blind judge"],
).reindex(index=["CORRECT", "INCORRECT"], columns=["CORRECT", "INCORRECT"], fill_value=0)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(confusion.values, cmap="Blues")
ax.set_xticks(range(2), confusion.columns)
ax.set_yticks(range(2), confusion.index)
ax.set_xlabel("Blind self-judge verdict")
ax.set_ylabel("Gold-aware judge verdict")
ax.set_title("Blind self-judge mostly over-approves")
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        value = confusion.iloc[i, j]
        share = value / len(df)
        color = "white" if value > confusion.values.max() * 0.55 else "#222222"
        ax.text(j, i, f"{value}\n({share:.0%})", ha="center", va="center", color=color)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cases")
plt.tight_layout()
plt.savefig("plots/judge_confusion.png", dpi=120)
print("Saved plots/judge_confusion.png")

# ---------- 4. Confidence bins: actual correctness vs self-approval ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
x = list(range(len(cal)))
width = 0.36
actual = ax.bar(
    [i - width / 2 for i in x],
    cal["accuracy"],
    width,
    color="#4C78A8",
    label="Gold says CORRECT",
)
blind = ax.bar(
    [i + width / 2 for i in x],
    cal["blind_correct_rate"],
    width,
    color="#E45756",
    label="Blind judge says CORRECT",
)
annotate_bars(ax, actual, [f"{v:.0%}" for v in cal["accuracy"]], y_offset=0.015)
annotate_bars(ax, blind, [f"{v:.0%}" for v in cal["blind_correct_rate"]], y_offset=0.015)
ax.set_xticks(x, cal.index.astype(str))
ax.set_xlabel("Self-reported confidence bin")
ax.set_ylabel("Share of answers")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, 1.08)
ax.set_title("Self-judgement rises faster than actual correctness")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("plots/confidence_vs_judges.png", dpi=120)
print("Saved plots/confidence_vs_judges.png")

# ---------- 5. Reproducibility across runs ----------
def correct_count_by_question(verdict_col):
    per_q = df.groupby("q_id")[verdict_col].apply(lambda s: (s == "CORRECT").sum())
    return per_q.value_counts().reindex([0, 1, 2, 3], fill_value=0)


gold_counts = correct_count_by_question("gold_judge")
blind_counts = correct_count_by_question("blind_judge")
plot_counts = pd.DataFrame({"Gold judge": gold_counts, "Blind judge": blind_counts})

fig, ax = plt.subplots(figsize=(7, 4.5))
plot_counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#E45756"], width=0.72)
ax.set_xlabel("Number of CORRECT verdicts across 3 runs for the same question")
ax.set_ylabel("Number of questions")
ax.set_title("Reruns reveal which questions are unstable")
ax.set_xticklabels(["0/3", "1/3", "2/3", "3/3"], rotation=0)
ax.legend(frameon=False)
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=9)
ax.set_ylim(0, max(plot_counts.max()) + 5)
plt.tight_layout()
plt.savefig("plots/reproducibility.png", dpi=120)
print("Saved plots/reproducibility.png")

# ---------- 6. High-confidence failure examples ----------
failures = df_c[(df_c["confidence"] >= 80) & (df_c["gold_judge"] == "INCORRECT")].copy()
failures = failures.sort_values(["confidence", "q_id"], ascending=[False, True])
failures[["q_id", "run", "confidence", "question", "gold", "answer", "blind_judge"]].head(10).to_csv(
    "plots/high_confidence_failures.csv",
    index=False,
)
print("Saved plots/high_confidence_failures.csv")

# ---------- 7. Self-vs-judge agreement ----------
high_conf = df_c[df_c["confidence"] >= 80]
if len(high_conf):
    rate = (high_conf["blind_judge"] == "CORRECT").mean()
    print(f"\n=== When confidence >= 80, blind judge says CORRECT in "
          f"{rate:.1%} of {len(high_conf)} cases ===")

# ---------- 8. Reproducibility across runs ----------
print("\n=== Reproducibility across the 3 runs ===")
per_q = df.groupby("q_id")["gold_judge"].nunique()
all_agree = (per_q == 1).mean()
print(f"Questions where all 3 runs got the same gold-judge verdict: {all_agree:.1%}")

per_q_blind = df.groupby("q_id")["blind_judge"].nunique()
all_agree_b = (per_q_blind == 1).mean()
print(f"Questions where all 3 runs got the same blind-judge verdict: {all_agree_b:.1%}")
