"""
VLM (LIV) reward vs. environment success analysis.

Reads the saved rollout mp4s in an experiment folder. Each mp4 is a stitched
two-row video produced by `robocasa_experiment.py`:

    +-----------------+-----------------+-----------------+
    |  agent_left     |  agent_right    |  eye_in_hand    |   <- top    (real env)
    +-----------------+-----------------+-----------------+
    |  pred_view_2    |  pred_view_3    |  pred_view_1    |   <- bottom (video-UNet output)
    +-----------------+-----------------+-----------------+

We score the BOTTOM row (the generated video) with LIV against the task language
prompt, aggregate per rollout, and report:
  * Spearman / Pearson correlation with success
  * AUROC for separating success vs failure
  * Calibration: success rate per reward quartile

Each chunk of generated frames is repeated for `action_horizon` env steps inside
the mp4, so we sample frames every `action_horizon` to deduplicate.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve

from liv import load_liv
from liv.models.clip import clip as liv_clip
from torchvision import transforms as T


# task name -> language prompt (best-effort static map; robocasa actually
# samples per-episode lang in get_ep_meta()["lang"], we use a representative one).
TASK_PROMPT = {
    "CloseDoubleDoor":   "close the double door",
    "CloseDrawer":       "close the drawer",
    "CoffeePressButton": "press the button on the coffee machine",
    "CoffeeServeMug":    "place the mug under the coffee machine spout",
    "CoffeeSetupMug":    "place the mug into the coffee machine",
    "OpenDrawer":        "open the drawer",
    "OpenSingleDoor":    "open the single door",
    "PnPCabToCounter":   "move the object from the cabinet to the counter",
    "PnPCounterToCab":   "move the object from the counter to the cabinet",
    "PnPCounterToMicrowave": "move the object from the counter to the microwave",
    "PnPCounterToSink":   "move the object from the counter to the sink",
    "PnPMicrowaveToCounter": "move the object from the microwave to the counter",
    "PnPSinkToCounter":   "move the object from the sink to the counter",
    "PnPStoveToCounter":  "move the object from the stove to the counter",
    "TurnOffMicrowave":  "press the stop button on the microwave",
    "TurnOnMicrowave":   "press the start button on the microwave",
    "TurnOffStove":      "turn off the stove",
    "TurnOnStove":       "turn on the stove",
    "TurnOnSinkFaucet":  "turn on the sink faucet",
    "TurnOffSinkFaucet": "turn off the sink faucet",
    "TurnSinkSpout":     "turn the sink spout",
}


def prompt_for(task: str) -> str:
    if task in TASK_PROMPT:
        return TASK_PROMPT[task]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower().strip()


def parse_video_filename(p: Path):
    m = re.match(r"^(.*)_demo_(\d+)\.mp4$", p.name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def index_rollouts(exp_dir: Path):
    rec = exp_dir / "multi_environment_experiment_record.json"
    data = json.load(open(rec))
    rows = []
    for task, det in data["environments"].items():
        for demo, v in det["experiments"].items():
            if v.get("status") != "done":
                continue
            vp = exp_dir / f"{task}_{demo}.mp4"
            if not vp.exists():
                continue
            rows.append({
                "task": task,
                "demo": demo,
                "success": int(v["success"]),
                "video": str(vp),
            })
    return pd.DataFrame(rows)


def load_predicted_frames(video_path: str, action_horizon: int = 16, max_chunks: int = 64):
    """
    Returns list of PIL.Image (RGB) extracted from the BOTTOM half of the mp4,
    one frame per generation chunk (every `action_horizon` env steps).
    """
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if n <= 0:
        cap.release()
        return []

    # one representative frame per chunk: midpoint of each [k*ah, (k+1)*ah) window
    chunk_centers = list(range(action_horizon // 2, n, action_horizon))[:max_chunks]
    frames = []
    for idx in chunk_centers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            continue
        bottom = f[H // 2:, :, :]                 # generated video row
        rgb = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


# LIV preprocess: resize 224, center crop, ToTensor, mean/std (standard CLIP).
LIV_PRE = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
])


@torch.inference_mode()
def score_rollout(liv_model, video_path, prompt, device, action_horizon=16):
    frames = load_predicted_frames(video_path, action_horizon=action_horizon)
    if not frames:
        return float("nan"), float("nan"), 0
    imgs = torch.stack([LIV_PRE(f) for f in frames]).to(device)  # already in [0,1]
    tokens = liv_clip.tokenize([prompt]).to(device)

    img_emb = liv_model(input=imgs, modality="vision")
    txt_emb = liv_model(input=tokens, modality="text")
    core = liv_model.module if hasattr(liv_model, "module") else liv_model
    # broadcast: replicate text emb to match image batch
    txt_emb_b = txt_emb.expand(img_emb.shape[0], -1)
    sims = core.sim(img_emb, txt_emb_b).view(-1).float().cpu().numpy()
    return float(sims.mean()), float(sims[-min(3, len(sims)):].mean()), len(sims)


def calibration_table(df, col, n_bins=4):
    s = df.dropna(subset=[col]).copy()
    try:
        s["q"] = pd.qcut(s[col], n_bins, labels=False, duplicates="drop")
    except ValueError:
        s["q"] = 0
    cal = s.groupby("q").agg(
        n=("success", "size"),
        succ_rate=("success", "mean"),
        r_mean=(col, "mean"),
        r_min=(col, "min"),
        r_max=(col, "max"),
    ).reset_index()
    return cal


def report(df, col, out_dir: Path, label: str):
    s = df.dropna(subset=[col])
    y = s["success"].values
    r = s[col].values
    if len(set(y)) < 2:
        print(f"[{label}] only one class present (n={len(y)}, success={int(y.sum())}); skipping ROC/AUROC")
        sp = spearmanr(r, y).correlation if len(r) > 1 else float("nan")
        pe = pearsonr(r, y).statistic if len(r) > 1 else float("nan")
        auc = float("nan")
    else:
        sp = spearmanr(r, y).correlation
        pe = pearsonr(r, y).statistic
        auc = roc_auc_score(y, r)
        fpr, tpr, _ = roc_curve(y, r)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", c="gray", lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC ({label})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_{label}.png", dpi=140)
        plt.close()

    cal = calibration_table(s, col)
    cal.to_csv(out_dir / f"calibration_{label}.csv", index=False)
    plt.figure(figsize=(5, 3.5))
    plt.bar(cal["q"].astype(int), cal["succ_rate"], color="#3274A1")
    for i, row in cal.iterrows():
        plt.text(int(row["q"]), row["succ_rate"] + 0.02, f"{row['succ_rate']:.2f}\nn={int(row['n'])}",
                 ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.05)
    plt.xlabel(f"reward quartile ({label})")
    plt.ylabel("success rate")
    plt.title(f"Calibration ({label})")
    plt.tight_layout()
    plt.savefig(out_dir / f"calibration_{label}.png", dpi=140)
    plt.close()

    print(f"\n=== [{label}] n={len(s)} succ={int(y.sum())}/{len(y)} ===")
    print(f"  Spearman = {sp:.4f}")
    print(f"  Pearson  = {pe:.4f}")
    print(f"  AUROC    = {auc:.4f}")
    print(cal.to_string(index=False))
    return {"label": label, "n": int(len(s)), "n_success": int(y.sum()),
            "spearman": sp, "pearson": pe, "auroc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=None, type=Path,
                    help="defaults to <exp_dir>/_vlm_reward_analysis")
    ap.add_argument("--action_horizon", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rollouts")
    ap.add_argument("--save_debug_frame", action="store_true",
                    help="dump one cropped predicted frame for sanity-checking")
    args = ap.parse_args()

    exp_dir = args.exp_dir.resolve()
    out_dir = (args.out_dir or (exp_dir / "_vlm_reward_analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = index_rollouts(exp_dir)
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"Indexed {len(df)} done rollouts in {exp_dir.name} "
          f"(success={int(df.success.sum())}/{len(df)})")
    if len(df) == 0:
        sys.exit("No rollouts found.")

    # Optional: save a debug crop so you can verify "bottom = predicted".
    if args.save_debug_frame:
        sample = df.iloc[0]
        frames = load_predicted_frames(sample["video"], action_horizon=args.action_horizon)
        if frames:
            (out_dir / "_debug").mkdir(exist_ok=True)
            frames[len(frames) // 2].save(out_dir / "_debug" / f"{sample['task']}_{sample['demo']}_pred.png")
            print(f"Saved debug crop -> {out_dir/'_debug'}")

    print(f"Loading LIV on {args.device} ...")
    liv_model = load_liv()
    liv_model.eval()
    liv_model.to(args.device)
    # mirror diffusion.py: keep the inner .device flag in sync
    dev_str = "cpu" if args.device == "cpu" else f"cuda:{torch.cuda.current_device()}"
    for obj in (liv_model, getattr(liv_model, "module", None)):
        if obj is None:
            continue
        try:
            setattr(obj, "device", dev_str)
        except Exception:
            pass

    r_mean, r_lastk, n_chunks = [], [], []
    for i, row in df.reset_index(drop=True).iterrows():
        try:
            m, l, nc = score_rollout(liv_model, row["video"], prompt_for(row["task"]),
                                     device=args.device, action_horizon=args.action_horizon)
        except Exception as e:
            print(f"  [{i}] {row['task']}_{row['demo']}: scoring failed ({e})")
            m, l, nc = float("nan"), float("nan"), 0
        r_mean.append(m); r_lastk.append(l); n_chunks.append(nc)
        if i % 5 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] {row['task']}_{row['demo']} "
                  f"success={row['success']} chunks={nc} mean={m:.4f} lastk={l:.4f}")

    df["liv_mean"] = r_mean
    df["liv_lastk"] = r_lastk
    df["n_chunks"] = n_chunks
    df["prompt"] = df["task"].map(prompt_for)

    csv_path = out_dir / "rollouts.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-rollout scores -> {csv_path}")

    summary = []
    summary.append(report(df, "liv_mean", out_dir, "liv_mean"))
    summary.append(report(df, "liv_lastk", out_dir, "liv_lastk"))
    pd.DataFrame(summary).to_csv(out_dir / "metrics_summary.csv", index=False)

    # Per-task table for the mean reward
    per_task = (df.dropna(subset=["liv_mean"])
                .groupby("task")
                .agg(n=("success", "size"),
                     n_success=("success", "sum"),
                     succ_rate=("success", "mean"),
                     liv_mean=("liv_mean", "mean"),
                     liv_lastk=("liv_lastk", "mean"))
                .reset_index()
                .sort_values("succ_rate", ascending=False))
    per_task.to_csv(out_dir / "per_task.csv", index=False)
    print(f"\nPer-task summary -> {out_dir/'per_task.csv'}")
    print(per_task.to_string(index=False))


if __name__ == "__main__":
    main()
