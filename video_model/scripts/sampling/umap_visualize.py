"""
UMAP visualization of decoder bottleneck features.

Loads one or more .npz files produced by `robocasa_planner_features.py
--extract_features`, fits a single UMAP across the union, and produces a scatter
plot colored by task category, with per-model marker shapes when multiple .npz
files are provided.

Example:
    python scripts/sampling/umap_visualize.py \
        experiments/<log>/features_ours.npz \
        experiments/<log>/features_videopolicy.npz \
        --out experiments/<log>/umap.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import umap

CATEGORY_ORDER = [
    "Pick and Place",
    "Doors",
    "Drawers",
    "Twisting Knobs",
    "Turning Levers",
    "Pressing Buttons",
    "Insertion",
    "Unknown",
]


def load_npz(path):
    data = np.load(path, allow_pickle=False)
    return {
        "features": data["features"],
        "task_names": data["task_names"].astype(str),
        "categories": data["categories"].astype(str),
        "demo_ids": data["demo_ids"].astype(str),
        "model_ids": data["model_ids"].astype(str),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", nargs="+", help="One or more feature .npz files.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    feats, cats, models = [], [], []
    for p in args.npz_paths:
        d = load_npz(p)
        feats.append(d["features"])
        cats.append(d["categories"])
        models.append(d["model_ids"])
    features = np.concatenate(feats, axis=0)
    categories = np.concatenate(cats, axis=0)
    model_ids = np.concatenate(models, axis=0)

    print(f"Fitting UMAP on {features.shape[0]} samples, dim={features.shape[1]}")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    emb = reducer.fit_transform(features)

    unique_cats = [c for c in CATEGORY_ORDER if c in set(categories)]
    cmap = plt.get_cmap("tab10")
    color_for = {c: cmap(i % 10) for i, c in enumerate(unique_cats)}

    unique_models = sorted(set(model_ids.tolist()))
    markers = ["o", "^", "s", "D", "P", "X"]
    marker_for = {m: markers[i % len(markers)] for i, m in enumerate(unique_models)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for cat in unique_cats:
        for m in unique_models:
            mask = (categories == cat) & (model_ids == m)
            if not np.any(mask):
                continue
            label = cat if len(unique_models) == 1 else f"{cat} [{m}]"
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[color_for[cat]],
                marker=marker_for[m],
                s=36, alpha=0.8, edgecolors="none",
                label=label,
            )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Decoder bottleneck features (UMAP)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
