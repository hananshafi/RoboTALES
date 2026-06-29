"""Analyze LIV critic vs environment success.

Reads critic_records.jsonl produced by robocasa_experiment_critic.py and reports:
  - Spearman + Pearson(reward, success) for each aggregation
  - AUROC distinguishing success vs failure
  - Calibration table: success rate per reward quartile
  - Per-subtask subgoal-progress correlations

Usage:
  python scripts/sampling/critic_analysis.py \
      experiments/<log_folder>_critic/critic_records.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np


REWARD_KEYS = [
    "reward_task_mean",
    "reward_task_max",
    "reward_task_last_step_mean",
    "reward_task_last_frame",
    "reward_task_last_quarter_mean",
]


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average rank for ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv] + 1.0


def _auroc(scores, labels):
    """Mann-Whitney U based AUROC. labels in {0,1}."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    rsum_pos = ranks[: len(pos)].sum()
    u = rsum_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _quartile_calibration(scores, labels, n_bins=4):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(s) == 0:
        return []
    # quantile bin edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(s, qs)
    # ensure monotonic & extend last edge
    edges[-1] = edges[-1] + 1e-9
    bins = np.digitize(s, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b + 1, "n": 0, "lo": float("nan"),
                         "hi": float("nan"), "success_rate": float("nan"),
                         "mean_reward": float("nan")})
            continue
        rows.append({
            "bin": b + 1,
            "n": n,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "success_rate": float(y[mask].mean()),
            "mean_reward": float(s[mask].mean()),
        })
    return rows


def load_records(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="path to critic_records.jsonl")
    ap.add_argument("--per-task", action="store_true",
                    help="also report metrics broken down by task")
    args = ap.parse_args()

    rows = load_records(args.records)
    if not rows:
        print("no records found")
        sys.exit(1)

    succ = np.array([r["success"] for r in rows], dtype=int)
    n = len(rows)
    print(f"\n=== {n} rollouts | overall success: {succ.mean():.3f} ===\n")

    # 1) reward vs success
    print(f"{'aggregation':28s}  {'spearman':>9s}  {'pearson':>9s}  {'auroc':>7s}")
    print("-" * 60)
    for k in REWARD_KEYS:
        rewards = np.array([r.get(k, float("nan")) for r in rows], dtype=float)
        mask = ~np.isnan(rewards)
        sp = _spearman(rewards[mask], succ[mask])
        pe = _pearson(rewards[mask], succ[mask])
        au = _auroc(rewards[mask], succ[mask])
        print(f"{k:28s}  {sp:>9.3f}  {pe:>9.3f}  {au:>7.3f}")

    # 2) calibration on best-AUROC aggregation
    aurocs = {}
    for k in REWARD_KEYS:
        rewards = np.array([r.get(k, float("nan")) for r in rows], dtype=float)
        mask = ~np.isnan(rewards)
        aurocs[k] = _auroc(rewards[mask], succ[mask])
    best_key = max(aurocs, key=lambda k: -1 if np.isnan(aurocs[k]) else aurocs[k])
    print(f"\n--- calibration by quartile of {best_key} ---")
    rewards = np.array([r[best_key] for r in rows], dtype=float)
    table = _quartile_calibration(rewards, succ, n_bins=4)
    print(f"{'bin':>3s}  {'n':>4s}  {'lo':>7s}  {'hi':>7s}  {'mean_r':>7s}  {'succ_rate':>9s}")
    for row in table:
        print(f"{row['bin']:>3d}  {row['n']:>4d}  {row['lo']:>7.3f}  "
              f"{row['hi']:>7.3f}  {row['mean_reward']:>7.3f}  {row['success_rate']:>9.3f}")

    # 3) subgoal progress
    sg = np.array([r.get("subgoal_progress_frac", float("nan")) for r in rows], dtype=float)
    mask = ~np.isnan(sg)
    if mask.sum() > 1:
        print("\n--- subgoal progress (fraction of plan reached) ---")
        print(f"  spearman vs success: {_spearman(sg[mask], succ[mask]):.3f}")
        print(f"  pearson  vs success: {_pearson(sg[mask], succ[mask]):.3f}")
        print(f"  auroc                : {_auroc(sg[mask], succ[mask]):.3f}")
        print(f"  mean fraction reached: succ={sg[(mask) & (succ==1)].mean():.3f} "
              f"fail={sg[(mask) & (succ==0)].mean():.3f}")

    # 4) per-task breakdown
    if args.per_task:
        print("\n--- per-task ---")
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        print(f"{'task':40s}  {'n':>3s}  {'succ':>5s}  "
              f"{'sp(mean)':>9s}  {'auroc':>7s}")
        for task, rs in sorted(by_task.items()):
            y = np.array([x["success"] for x in rs], dtype=int)
            x = np.array([x["reward_task_mean"] for x in rs], dtype=float)
            sp = _spearman(x, y) if len(rs) > 1 else float("nan")
            au = _auroc(x, y)
            print(f"{task[:40]:40s}  {len(rs):>3d}  {y.mean():>5.2f}  "
                  f"{sp:>9.3f}  {au:>7.3f}")


if __name__ == "__main__":
    main()
