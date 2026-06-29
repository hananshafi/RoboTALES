"""Plot critic reward vs environment success.

Reads critic_records.jsonl produced by robocasa_experiment_critic.py and writes
PNGs into <records_dir>/critic_plots/:
  - roc.png             ROC curve per reward aggregation
  - calibration.png     success rate by reward quartile (one panel per agg)
  - reward_dist.png     reward distributions split by success / failure
  - metrics_summary.png Spearman / Pearson / AUROC bars per aggregation
  - per_task_auroc.png  per-task AUROC + success-rate bar chart

Usage:
  python scripts/sampling/critic_plots.py \
      experiments/<log_folder>_critic/critic_records.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Palette matched to the planner-vs-prompt figure: medium blue + muted purple.
PALETTE_BLUE = "#2c7fb8"
PALETTE_PURPLE = "#7d51a1"


REWARD_KEYS = [
    "reward_task_mean",
    "reward_task_max",
    "reward_task_last_step_mean",
    "reward_task_last_frame",
    "reward_task_last_quarter_mean",
]


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv] + 1.0


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2: return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0: return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _spearman(x, y):
    if len(x) < 2: return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _roc_curve(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = np.argsort(-s)
    y_sorted = y[order]
    P = max(int(y.sum()), 1); N = max(int((1 - y).sum()), 1)
    tpr = np.concatenate([[0.0], np.cumsum(y_sorted) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y_sorted) / N])
    return fpr, tpr


def _quartile_calibration(scores, labels, n_bins=4):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = np.digitize(s, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        m = bins == b
        n = int(m.sum())
        rows.append({
            "bin": b + 1, "n": n,
            "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "success_rate": float(y[m].mean()) if n else float("nan"),
            "mean_reward": float(s[m].mean()) if n else float("nan"),
        })
    return rows


def load_records(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_fig(fig, out_path, dpi=160):
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    if out_path.lower().endswith(".png"):
        pdf_path = out_path[:-4] + ".pdf"
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)


def plot_roc(rows, succ, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for k in REWARD_KEYS:
        r = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        m = ~np.isnan(r)
        if m.sum() < 2: continue
        fpr, tpr = _roc_curve(r[m], succ[m])
        au = _auroc(r[m], succ[m])
        ax.plot(fpr, tpr, label=f"{k} (AUC={au:.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC: critic reward vs environment success")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_calibration(rows, succ, out_path, n_bins=4):
    fig, axes = plt.subplots(1, len(REWARD_KEYS),
                             figsize=(4 * len(REWARD_KEYS), 4), sharey=True)
    overall = succ.mean()
    for ax, k in zip(axes, REWARD_KEYS):
        r = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        m = ~np.isnan(r)
        if m.sum() < n_bins:
            ax.set_title(f"{k}\n(insufficient data)"); continue
        table = _quartile_calibration(r[m], succ[m], n_bins=n_bins)
        x = [t["bin"] for t in table]
        rates = [t["success_rate"] for t in table]
        ns = [t["n"] for t in table]
        bars = ax.bar(x, rates, color="steelblue", edgecolor="black")
        for b, n_, rate in zip(bars, ns, rates):
            if not np.isnan(rate):
                ax.text(b.get_x() + b.get_width() / 2, rate + 0.02,
                        f"n={n_}\n{rate:.2f}", ha="center", va="bottom",
                        fontsize=8)
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1,
                   label=f"overall={overall:.2f}")
        ax.set_xticks(x); ax.set_xticklabels([f"Q{i}" for i in x])
        ax.set_xlabel("reward quartile (low → high)")
        ax.set_title(k, fontsize=9); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("success rate")
    fig.suptitle("Calibration: success rate by critic-reward quartile", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_reward_dist(rows, succ, out_path):
    fig, axes = plt.subplots(1, len(REWARD_KEYS),
                             figsize=(3.6 * len(REWARD_KEYS), 4))
    rng = np.random.RandomState(0)
    for ax, k in zip(axes, REWARD_KEYS):
        r = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        m = ~np.isnan(r)
        groups = [r[m & (succ == 0)], r[m & (succ == 1)]]
        positions = [1, 2]
        labels = [f"fail (n={len(groups[0])})", f"success (n={len(groups[1])})"]
        if all(len(g) > 0 for g in groups):
            parts = ax.violinplot(groups, positions=positions,
                                  showmeans=False, showmedians=True, widths=0.8)
            for body, color in zip(parts["bodies"], ["#cc4d4d", "#4d8acc"]):
                body.set_facecolor(color); body.set_alpha(0.55)
        for pos, g, c in zip(positions, groups, ["#7a1f1f", "#1f3f7a"]):
            jitter = (rng.rand(len(g)) - 0.5) * 0.18
            ax.scatter(pos + jitter, g, s=6, alpha=0.45, color=c)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(k, fontsize=9); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("critic reward")
    fig.suptitle("Critic reward distribution by outcome", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(rows, succ, out_path):
    metrics = {"spearman": [], "pearson": [], "auroc": []}
    for k in REWARD_KEYS:
        r = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        m = ~np.isnan(r)
        metrics["spearman"].append(_spearman(r[m], succ[m]))
        metrics["pearson"].append(_pearson(r[m], succ[m]))
        metrics["auroc"].append(_auroc(r[m], succ[m]))

    sg = np.array([row.get("subgoal_progress_frac", np.nan) for row in rows], dtype=float)
    sg_m = ~np.isnan(sg)
    keys = list(REWARD_KEYS)
    if sg_m.sum() > 1:
        keys.append("subgoal_progress_frac")
        metrics["spearman"].append(_spearman(sg[sg_m], succ[sg_m]))
        metrics["pearson"].append(_pearson(sg[sg_m], succ[sg_m]))
        metrics["auroc"].append(_auroc(sg[sg_m], succ[sg_m]))

    x = np.arange(len(keys)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(keys)), 5))
    ax.bar(x - w, metrics["spearman"], w, label="Spearman", color="#4d8acc")
    ax.bar(x,     metrics["pearson"],  w, label="Pearson",  color="#cc8a4d")
    ax.bar(x + w, metrics["auroc"],    w, label="AUROC",    color="#4dcc7a")
    for off, vs in [(-w, metrics["spearman"]), (0, metrics["pearson"]),
                    (w, metrics["auroc"])]:
        for xi, v in zip(x + off, vs):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
               label="AUROC chance")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title("Critic reward → success: rank / linear / ranking metrics")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_roc_pair(rows, succ, out_path,
                  good_key="reward_task_mean", bad_key="reward_task_max"):
    """Two-curve ROC: trajectory-averaged reward vs single-frame max.

    Shows the signal is sustained agreement, not a spurious frame-level peak.
    Appendix styling: blue/purple palette, compact fonts.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    palette = {good_key: PALETTE_BLUE, bad_key: PALETTE_PURPLE}
    for k in (good_key, bad_key):
        r = np.array([row.get(k, np.nan) for row in rows], dtype=float)
        m = ~np.isnan(r)
        if m.sum() < 2:
            continue
        fpr, tpr = _roc_curve(r[m], succ[m])
        au = _auroc(r[m], succ[m])
        short = "mean" if k == good_key else "max"
        ax.plot(fpr, tpr,
                label=f"{short}  (AUC={au:.2f})",
                linewidth=2.0, color=palette[k])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, linewidth=1.0,
            label="chance")
    ax.set_xlabel("False positive rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=13, fontweight="bold")
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    ax.set_title("ROC: reward vs success", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10, loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_path, dpi=160)
    plt.close(fig)


def plot_calibration_single(rows, succ, out_path, agg_key="reward_task_mean",
                            n_bins=4):
    """Single-panel calibration: success rate by quartile of one aggregation.

    Appendix styling: image blue/purple palette, compact fonts.
    """
    r = np.array([row.get(agg_key, np.nan) for row in rows], dtype=float)
    m = ~np.isnan(r)
    table = _quartile_calibration(r[m], succ[m], n_bins=n_bins)
    overall = float(succ.mean())
    auroc = _auroc(r[m], succ[m])
    spearman = _spearman(r[m], succ[m])

    x = [t["bin"] for t in table]
    rates = [t["success_rate"] for t in table]
    ns = [t["n"] for t in table]
    mean_r = [t["mean_reward"] for t in table]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, rates, color=PALETTE_BLUE, edgecolor="black", width=0.72)
    for b, n_, rate, mr in zip(bars, ns, rates, mean_r):
        ax.text(b.get_x() + b.get_width() / 2, rate + 0.018,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.02,
                f"n={n_}\nr̄={mr:.2f}", ha="center", va="bottom", fontsize=9,
                color="white")
    ax.axhline(overall, color="gray", linestyle="--", linewidth=1.0,
               label=f"overall = {overall:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in x], fontsize=12)
    ax.set_xlabel("reward quartile  (low → high)", fontsize=13, fontweight="bold")
    ax.set_ylabel("success rate", fontsize=13, fontweight="bold")
    ax.xaxis.labelpad = 6
    ax.yaxis.labelpad = 6
    ax.set_title(
        f"Calibration  (AUROC={auroc:.2f},  N={int(m.sum())})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_path, dpi=160)
    plt.close(fig)


def plot_per_task(rows, out_path, agg_key="reward_task_last_frame"):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks, n, succ_rate, aurocs = [], [], [], []
    for t, rs in sorted(by_task.items()):
        y = np.array([x["success"] for x in rs], dtype=int)
        s = np.array([x.get(agg_key, np.nan) for x in rs], dtype=float)
        m = ~np.isnan(s)
        tasks.append(t); n.append(len(rs))
        succ_rate.append(float(y.mean()))
        aurocs.append(_auroc(s[m], y[m]) if m.sum() else float("nan"))

    x = np.arange(len(tasks)); w = 0.42
    fig, ax = plt.subplots(figsize=(max(10, 0.42 * len(tasks)), 5))
    ax.bar(x - w / 2, succ_rate, w, label="success rate", color="#4d8acc")
    ax.bar(x + w / 2, aurocs,    w, label=f"AUROC ({agg_key})", color="#4dcc7a")
    for xi, v in zip(x - w / 2, succ_rate):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for xi, v in zip(x + w / 2, aurocs):
        if np.isnan(v):
            ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=6,
                    color="gray")
        else:
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1); ax.set_ylabel("rate")
    ax.set_title("Per-task: success rate vs critic AUROC "
                 "(AUROC undefined when all rollouts succeed/fail)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="path to critic_records.jsonl")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <records_dir>/critic_plots)")
    args = ap.parse_args()

    rows = load_records(args.records)
    if not rows:
        print("no records found"); sys.exit(1)
    succ = np.array([r["success"] for r in rows], dtype=int)

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.records)), "critic_plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{len(rows)} rollouts | overall success = {succ.mean():.3f}")
    print(f"writing plots to {out_dir}")

    plot_roc(rows, succ, os.path.join(out_dir, "roc.png"))
    plot_roc_pair(rows, succ, os.path.join(out_dir, "roc_mean_vs_max.png"))
    plot_calibration(rows, succ, os.path.join(out_dir, "calibration.png"))
    plot_calibration_single(rows, succ,
                            os.path.join(out_dir, "calibration_mean.png"),
                            agg_key="reward_task_mean")
    plot_reward_dist(rows, succ, os.path.join(out_dir, "reward_dist.png"))
    plot_metrics_summary(rows, succ, os.path.join(out_dir, "metrics_summary.png"))
    plot_per_task(rows, os.path.join(out_dir, "per_task_auroc.png"))

    print("done.")


if __name__ == "__main__":
    main()
