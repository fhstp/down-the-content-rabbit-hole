"""tune_clip_hyperparams.py — Grid-search hyperparameter tuning for classify_clip.py.

Optimises the sliding-window aggregation hyperparameters of classify_clip.py
with respect to F1 score on the annotated subset of the dataset.

All frame embeddings are loaded from the classify_clip.py embedding cache;
no video decoding or CLIP image-encoding is needed if the cache is warm.

Algorithm
---------
1.  Load CLIP model and encode text prompt banks once per class.
2.  Load per-frame embeddings from the .npz cache for every annotated video.
3.  Pre-compute, per (video, class, frame), the max cosine-similarity across
    the 3 spatial crops for every positive and negative prompt.  The per-prompt
    similarity vectors are stored sorted in descending order so any top-K mean
    can be evaluated in O(1) without repeating the embedding arithmetic.
4.  Grid-search over (top_k, per_frame_cutoff, window_size, min_in_window,
    threshold / count_threshold).  For each outer (top_k, class) the entire
    pos_prob matrix (n_videos × max_frames) is built with a single batched
    softmax, then all threshold combinations are evaluated with a vectorised
    numpy rolling-sum pass — no Python loops over videos or frames.
5.  Report top-N results per class and write:
      • tune_results.json  — full sorted grid results
      • best_params.json   — one best record per class, ready for classify_clip.py

Outputs
-------
Terminal:        per-class top-N table sorted by F1.
tune_results.json:    { class_name: [ {params + metrics}, ... ] }
best_params.json:     { class_name: {top_k, per_frame_cutoff, window_size,
                                     min_positives_in_window, mode,
                                     threshold | count_threshold,
                                     f1, precision, recall} }
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from tqdm.rich import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = str(REPO_ROOT / "dataset" / "yt_data.json")
DEFAULT_DATASET_ROOT = str(REPO_ROOT / "dataset")

# Re-use helpers from classify_clip.py and evaluate.py — no duplication.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_clip import (
    DEFAULT_GENERIC_PROMPTS,
    _embedding_cache_path,
    _sanitize_model_key,
    build_prompt_banks,
    load_cached_embeddings,
    load_video_frames,
    preprocess_frame,
    save_cached_embeddings,
)
from evaluate import load_annotated_data


# ── Vectorised F1 evaluation ──────────────────────────────────────────────────

def _rolling_active_counts(
    binary_mat: np.ndarray,    # (n_videos, max_frames) bool
    valid_mask: np.ndarray,    # (n_videos, max_frames) bool
    window_size: int,
    min_in_window: int,
) -> np.ndarray:               # (n_videos,) int32 — active frame count per video
    """Compute sliding-window active frame counts for all videos at once.

    A frame t is "active" if:
      • binary_mat[v, t] is True, AND
      • the rolling sum of binary_mat[v, t-W+1 : t+1] >= min_in_window, AND
      • valid_mask[v, t] is True (not a padding frame).

    Uses the cumsum trick for O(max_frames) computation per video with no
    Python loops over frames.
    """
    n_videos, max_frames = binary_mat.shape

    # Only real, positive frames contribute to the rolling sum.
    b = (binary_mat & valid_mask).astype(np.int32)                      # (n, F)

    cs = np.zeros((n_videos, max_frames + 1), dtype=np.int32)
    cs[:, 1:] = np.cumsum(b, axis=1)

    # rolling_sum[v, t] = sum of b[v, t-W+1 : t+1]
    starts       = np.maximum(0, np.arange(max_frames, dtype=np.int32) + 1 - window_size)
    rolling_sum  = cs[:, 1:] - cs[:, starts]                            # (n, F)

    active_mat   = (binary_mat & valid_mask) & (rolling_sum >= min_in_window)
    return active_mat.sum(axis=1).astype(np.int32)                      # (n,)


def vectorised_f1(
    pos_probs_mat: np.ndarray,         # (n_videos, max_frames) float32
    valid_mask: np.ndarray,            # (n_videos, max_frames) bool
    n_frames_vec: np.ndarray,          # (n_videos,) int32
    gt_vec: np.ndarray,                # (n_videos,) bool
    per_frame_cutoff: float,
    window_size: int,
    min_in_window: int,
    threshold_values: List[float],
    count_threshold_values: List[int],
) -> List[Dict]:
    """Evaluate every threshold value in one pass and return a list of metric dicts."""
    binary = (pos_probs_mat > per_frame_cutoff)                        # (n, F)
    active_counts = _rolling_active_counts(binary, valid_mask, window_size, min_in_window)

    results: List[Dict] = []

    # ── Relative thresholds (active_count / n_frames > threshold) ────────────
    safe_n = np.where(n_frames_vec > 0, n_frames_vec, 1).astype(np.float32)
    active_fractions = np.where(
        n_frames_vec > 0,
        active_counts.astype(np.float32) / safe_n,
        0.0,
    )

    for thr in threshold_values:
        predictions = active_fractions > thr
        tp = int(np.sum( predictions &  gt_vec))
        fp = int(np.sum( predictions & ~gt_vec))
        fn = int(np.sum(~predictions &  gt_vec))
        tn = int(np.sum(~predictions & ~gt_vec))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            "mode":      "relative",
            "threshold": thr,
            "f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # ── Absolute thresholds (active_count >= count_threshold) ────────────────
    for cnt_thr in count_threshold_values:
        predictions = active_counts >= cnt_thr
        tp = int(np.sum( predictions &  gt_vec))
        fp = int(np.sum( predictions & ~gt_vec))
        fn = int(np.sum(~predictions &  gt_vec))
        tn = int(np.sum(~predictions & ~gt_vec))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            "mode":            "absolute",
            "count_threshold": cnt_thr,
            "f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return results


# ── Pre-computation helpers ───────────────────────────────────────────────────

def build_sorted_sim_arrays(
    data: List[Dict],
    class_names: List[str],
    text_embeds_np: Dict[str, Dict[str, np.ndarray]],
    model_key: str,
    embedding_cache: str,
    fps: float,
    max_frames: int,
    preprocess_fn,
    model,
    device: torch.device,
) -> Tuple[
    Dict[str, np.ndarray],   # all_sorted_pos[cls]: (n, F, N_pos)
    Dict[str, np.ndarray],   # all_sorted_neg[cls]: (n, F, N_neg)
    np.ndarray,              # valid_mask:  (n, F) bool
    np.ndarray,              # n_frames_vec: (n,)  int32
]:
    """Load / encode frame embeddings and pre-compute per-prompt sorted cosine sims.

    Returns large numpy arrays indexed by (video_idx, frame_idx, prompt_idx).
    The prompt axis is sorted in descending cosine-similarity order so that
    mean([:k]) gives the top-K mean for any k with no extra computation.
    """
    n_videos = len(data)

    # Determine maximum frame count from cache metadata (avoid loading full arrays).
    # We do a quick first pass to discover frame counts and build the arrays.
    frame_counts: List[int] = [0] * n_videos
    embeds_per_video: List[Optional[List[np.ndarray]]] = [None] * n_videos
    missing = 0

    print("  Pass 1/2 — loading frame embeddings from cache...")
    for v_idx, entry in enumerate(tqdm(data, desc="load cache")):
        video_id   = entry["video_id"]
        cache_path = _embedding_cache_path(embedding_cache, model_key, fps, max_frames, video_id)
        cached     = load_cached_embeddings(cache_path)

        if cached is None:
            video_path = entry.get("absolute_path", "")
            if not os.path.exists(video_path):
                print(f"  Warning: no cache and no video for {video_id} — skipped")
                missing += 1
                continue
            try:
                frames, timestamps = load_video_frames(video_path, fps, max_frames)
                embeds_np_list: List[np.ndarray] = []
                with torch.no_grad():
                    for frame in frames:
                        pre   = preprocess_frame(frame, preprocess_fn, device)
                        embed = model.encode_image(pre).float().cpu().numpy()
                        embeds_np_list.append(embed)
                save_cached_embeddings(cache_path, embeds_np_list, timestamps)
                cached = (embeds_np_list, timestamps)
            except Exception as exc:
                print(f"  Error encoding {video_id}: {exc} — skipped")
                missing += 1
                continue

        embeds_list, _ = cached
        frame_counts[v_idx]    = len(embeds_list)
        embeds_per_video[v_idx] = embeds_list

    if missing:
        print(f"  Warning: {missing} videos had no cache or video file and were skipped.")

    n_frames_vec = np.array(frame_counts, dtype=np.int32)
    max_f        = int(n_frames_vec.max()) if n_frames_vec.max() > 0 else 1
    valid_mask   = np.zeros((n_videos, max_f), dtype=bool)

    # Pre-allocate sorted-sim arrays per class.
    all_sorted_pos: Dict[str, np.ndarray] = {}
    all_sorted_neg: Dict[str, np.ndarray] = {}
    for cls in class_names:
        N_pos = text_embeds_np[cls]["positive"].shape[0]
        N_neg = text_embeds_np[cls]["negative"].shape[0]
        all_sorted_pos[cls] = np.zeros((n_videos, max_f, N_pos), dtype=np.float32)
        all_sorted_neg[cls] = np.zeros((n_videos, max_f, N_neg), dtype=np.float32)

    # L2-normalise text embeddings once (re-used for every video).
    pos_n: Dict[str, np.ndarray] = {}
    neg_n: Dict[str, np.ndarray] = {}
    for cls in class_names:
        p = text_embeds_np[cls]["positive"].astype(np.float32)
        n = text_embeds_np[cls]["negative"].astype(np.float32)
        pos_n[cls] = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        neg_n[cls] = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)

    print("  Pass 2/2 — computing sorted cosine-similarity vectors...")
    for v_idx, embeds_list in enumerate(tqdm(embeds_per_video, desc="compute sims")):
        if embeds_list is None:
            continue
        nf = len(embeds_list)
        valid_mask[v_idx, :nf] = True

        # Stack all frames into (nf, 3, D) and L2-normalise.
        stack   = np.stack(embeds_list).astype(np.float32)              # (nf, 3, D)
        norms   = np.linalg.norm(stack, axis=-1, keepdims=True) + 1e-8
        stack_n = stack / norms                                          # (nf, 3, D)

        for cls in class_names:
            # Cosine sims via batched matmul: (nf, 3, D) @ (D, N_pos) → (nf, 3, N_pos)
            sims_pos = stack_n @ pos_n[cls].T                           # (nf, 3, N_pos)
            sims_neg = stack_n @ neg_n[cls].T                           # (nf, 3, N_neg)
            max_pos  = sims_pos.max(axis=1)                              # (nf, N_pos)
            max_neg  = sims_neg.max(axis=1)                              # (nf, N_neg)
            # Sort descending so [:k].mean() = top-K mean.
            sp = np.sort(max_pos, axis=1)[:, ::-1]                      # (nf, N_pos)
            sn = np.sort(max_neg, axis=1)[:, ::-1]                      # (nf, N_neg)
            all_sorted_pos[cls][v_idx, :nf, :] = sp
            all_sorted_neg[cls][v_idx, :nf, :] = sn

    return all_sorted_pos, all_sorted_neg, valid_mask, n_frames_vec


# ── Output helpers ────────────────────────────────────────────────────────────

def print_global_results(
    global_results: List[Dict],
    class_names: List[str],
    top_n: int,
) -> None:
    """Print top-N results for global (macro-averaged) tuning."""
    if not global_results:
        return
    rows    = global_results[:top_n]
    n_total = len(global_results)
    W = 114
    print(f"\n{'=' * W}")
    print(
        f"  Global (macro-averaged F1 across {len(class_names)} classes)"
        f"  —  top {min(top_n, n_total)} of {n_total} results, sorted by macro-F1"
    )
    print(f"{'=' * W}")
    hdr = (
        f"  {'macro-F1':>8} {'prec':>6} {'rec':>6}  "
        f"{'top_k':>6} {'cutoff':>7} {'ws':>4} {'minW':>5}  "
        f"{'mode':<9} {'thr/cnt':>8}"
    )
    print(hdr)
    print(f"  {'-' * (W - 2)}")
    for r in rows:
        mode_str  = "relative" if r["mode"] == "relative" else "absolute"
        thr_str   = f"{r['threshold']:.2f}" if r["mode"] == "relative" else str(r["count_threshold"])
        top_k_str = str(r["top_k"]) if r["top_k"] != -1 else "all"
        print(
            f"  {r['macro_f1']:>8.4f} {r['macro_precision']:>6.4f} {r['macro_recall']:>6.4f}  "
            f"{top_k_str:>6} {r['per_frame_cutoff']:>7.2f} "
            f"{r['window_size']:>4} {r['min_in_window']:>5}  "
            f"{mode_str:<9} {thr_str:>8}"
        )
    print(f"{'=' * W}")


def print_results(all_results: Dict[str, List[Dict]], class_names: List[str], top_n: int) -> None:
    W = 114
    for cls in class_names:
        rows = all_results[cls][:top_n]
        n_total = len(all_results[cls])
        print(f"\n{'=' * W}")
        print(f"  Class: {cls}  —  top {min(top_n, n_total)} of {n_total} results, sorted by F1")
        print(f"{'=' * W}")
        hdr = (
            f"  {'F1':>6} {'Prec':>6} {'Rec':>6}  "
            f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
            f"{'top_k':>6} {'cutoff':>7} {'ws':>4} {'minW':>5}  "
            f"{'mode':<9} {'thr/cnt':>8}"
        )
        print(hdr)
        print(f"  {'-' * (W - 2)}")
        for r in rows:
            if r["mode"] == "relative":
                mode_str = "relative"
                thr_str  = f"{r['threshold']:.2f}"
            else:
                mode_str = "absolute"
                thr_str  = str(r["count_threshold"])
            top_k_str = str(r["top_k"]) if r["top_k"] != -1 else "all"
            print(
                f"  {r['f1']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f}  "
                f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4}  "
                f"{top_k_str:>6} {r['per_frame_cutoff']:>7.2f} "
                f"{r['window_size']:>4} {r['min_in_window']:>5}  "
                f"{mode_str:<9} {thr_str:>8}"
            )
    print(f"\n{'=' * W}")


def save_outputs(
    all_results: Dict[str, List[Dict]],
    class_names: List[str],
    output_path: str,
    best_params_path: str,
    global_best_params_path: str = "",
) -> None:
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull grid results saved → {output_path}")

    best: Dict[str, Dict] = {}
    for cls in class_names:
        if not all_results[cls]:
            continue
        r = all_results[cls][0]
        entry: Dict = {
            "top_k":                   r["top_k"],
            "per_frame_cutoff":        r["per_frame_cutoff"],
            "window_size":             r["window_size"],
            "min_positives_in_window": r["min_in_window"],
            "mode":                    r["mode"],
            "f1":                      r["f1"],
            "precision":               r["precision"],
            "recall":                  r["recall"],
        }
        if r["mode"] == "relative":
            entry["threshold"] = r["threshold"]
        else:
            entry["count_threshold"] = r["count_threshold"]
        best[cls] = entry

    with open(best_params_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved       → {best_params_path}\n")
    print("Best params per class:")
    for cls, b in best.items():
        top_k_str = str(b["top_k"]) if b["top_k"] != -1 else "all"
        if b["mode"] == "relative":
            thr_info = f"threshold={b['threshold']}"
        else:
            thr_info = f"count_threshold={b['count_threshold']}"
        print(
            f"  {cls:<30}  F1={b['f1']:.4f}  top_k={top_k_str}  "
            f"cutoff={b['per_frame_cutoff']}  ws={b['window_size']}  "
            f"minW={b['min_positives_in_window']}  {thr_info}"
        )

    # ── Global best params ────────────────────────────────────────────────────
    if global_best_params_path and all_results.get("_global"):
        g = all_results["_global"][0]
        global_best: Dict = {
            "top_k":                   g["top_k"],
            "per_frame_cutoff":        g["per_frame_cutoff"],
            "window_size":             g["window_size"],
            "min_positives_in_window": g["min_in_window"],
            "mode":                    g["mode"],
            "macro_f1":                g["macro_f1"],
            "macro_precision":         g["macro_precision"],
            "macro_recall":            g["macro_recall"],
            "per_class_f1":            g["per_class_f1"],
        }
        if g["mode"] == "relative":
            global_best["threshold"] = g["threshold"]
        else:
            global_best["count_threshold"] = g["count_threshold"]
        with open(global_best_params_path, "w") as f:
            json.dump(global_best, f, indent=2)
        top_k_str = str(g["top_k"]) if g["top_k"] != -1 else "all"
        thr_info  = (
            f"threshold={g['threshold']}"
            if g["mode"] == "relative"
            else f"count_threshold={g['count_threshold']}"
        )
        print(f"\nGlobal best params saved → {global_best_params_path}")
        print(
            f"  macro-F1={g['macro_f1']:.4f}  top_k={top_k_str}  "
            f"cutoff={g['per_frame_cutoff']}  ws={g['window_size']}  "
            f"minW={g['min_in_window']}  {thr_info}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Parse grid values ────────────────────────────────────────────────────
    top_k_values           = [int(x)   for x in args.top_k_values.split(",")]
    cutoff_values          = [float(x) for x in args.per_frame_cutoff_values.split(",")]
    window_size_values     = [int(x)   for x in args.window_size_values.split(",")]
    min_pos_fractions      = [float(x) for x in args.min_positives_fractions.split(",")]
    threshold_values       = [float(x) for x in args.threshold_values.split(",")]
    count_threshold_values = (
        [int(x) for x in args.count_threshold_values.split(",")]
        if args.count_threshold_values.strip()
        else []
    )

    if not threshold_values and not count_threshold_values:
        raise SystemExit(
            "error: at least one of --threshold-values or --count-threshold-values "
            "must be non-empty."
        )

    # Unique (window_size, min_in_window) pairs — fractions × sizes with dedup.
    win_pairs: List[Tuple[int, int]] = sorted({
        (ws, max(1, round(frac * ws)))
        for ws in window_size_values
        for frac in min_pos_fractions
    })

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) != args.device:
        print(f"Warning: CUDA not available; falling back to {device}")

    # ── Classes ───────────────────────────────────────────────────────────────
    with open(args.classes_config) as f:
        classes_config = json.load(f)
    class_names = (
        [c.strip() for c in args.classes.split(",")]
        if args.classes
        else list(classes_config.keys())
    )

    # ── Annotated dataset ─────────────────────────────────────────────────────
    data      = load_annotated_data(args.dataset, args.dataset_root)
    n_videos  = len(data)
    video_ids = [e["video_id"] for e in data]
    print(f"Loaded {n_videos} annotated videos from {args.dataset}")

    gt_mat: Dict[str, np.ndarray] = {
        cls: np.array([cls in e.get("consensus_labels", []) for e in data], dtype=bool)
        for cls in class_names
    }
    print("Class distribution (annotated):")
    for cls in class_names:
        pos = int(gt_mat[cls].sum())
        print(f"  {cls:<30}  {pos:>4} positive  {n_videos - pos:>4} negative")

    # ── Prompt banks ─────────────────────────────────────────────────────────
    prompt_mode = args.prompt_mode
    if prompt_mode is None:
        prompt_mode = "config" if os.path.exists(args.prompt_config) else "auto"
        print(f"--prompt-mode not set; using '{prompt_mode}'")
    if prompt_mode == "llm":
        raise SystemExit("error: --prompt-mode llm is not supported in the tuning script.")

    prompt_banks = build_prompt_banks(
        class_names, classes_config, prompt_mode, args.prompt_config,
    )

    # ── Load CLIP model ───────────────────────────────────────────────────────
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    print(f"\nLoading CLIP {args.model} on {device}...")
    model, preprocess_fn = clip.load(args.model, device=device)
    model = model.float()
    model.eval()

    # ── Encode text embeddings ────────────────────────────────────────────────
    print("Encoding text prompt banks...")
    text_embeds_np: Dict[str, Dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for cls in class_names:
            pos_prompts = prompt_banks[cls]["positive"]
            neg_prompts = prompt_banks[cls]["negative"]
            pos_tok     = clip.tokenize(pos_prompts).to(device)
            neg_tok     = clip.tokenize(neg_prompts).to(device)
            text_embeds_np[cls] = {
                "positive": model.encode_text(pos_tok).float().cpu().numpy(),
                "negative": model.encode_text(neg_tok).float().cpu().numpy(),
            }
            print(f"  {cls}: {len(pos_prompts)} pos, {len(neg_prompts)} neg prompts")

    model_key = _sanitize_model_key(args.model)

    # ── Pre-compute sorted cosine-similarity arrays ───────────────────────────
    print("\nPre-computing sorted cosine-similarity arrays...")
    all_sorted_pos, all_sorted_neg, valid_mask, n_frames_vec = build_sorted_sim_arrays(
        data, class_names, text_embeds_np,
        model_key, args.embedding_cache, args.fps, args.max_frames,
        preprocess_fn, model, device,
    )

    # ── Grid search ───────────────────────────────────────────────────────────
    n_combos = (
        len(top_k_values) * len(class_names) * len(cutoff_values) * len(win_pairs)
        * (len(threshold_values) + len(count_threshold_values))
    )
    print(f"\nGrid search:")
    print(f"  top_k × classes × cutoffs × window_pairs × thresholds")
    print(
        f"  {len(top_k_values)} × {len(class_names)} × {len(cutoff_values)} "
        f"× {len(win_pairs)} × {len(threshold_values) + len(count_threshold_values)}"
        f" = {n_combos:,} total evaluations"
    )

    all_results: Dict[str, List[Dict]] = {cls: [] for cls in class_names}

    # Accumulates (cls, f1, precision, recall) per combo-key for global tuning.
    # combo_key = (top_k, cutoff, ws, min_in_w, mode, threshold_or_count)
    global_f1_accum: Dict[tuple, List] = defaultdict(list)

    for top_k in tqdm(top_k_values, desc="top_k"):
        # ── Build pos_probs_mat for all classes with one batched softmax ──────
        pos_probs_mats: Dict[str, np.ndarray] = {}
        for cls in class_names:
            sp = all_sorted_pos[cls]  # (n, F, N_pos)
            sn = all_sorted_neg[cls]  # (n, F, N_neg)
            N_pos = sp.shape[2]
            N_neg = sn.shape[2]
            k_pos = N_pos if top_k == -1 else min(top_k, N_pos)
            k_neg = N_neg if top_k == -1 else min(top_k, N_neg)

            mean_pos = sp[:, :, :k_pos].mean(axis=2)   # (n, F)
            mean_neg = sn[:, :, :k_neg].mean(axis=2)   # (n, F)

            # softmax(100 × [mean_pos, mean_neg])[0] — numerically stable
            x_pos  = (100.0 * mean_pos).astype(np.float64)
            x_neg  = (100.0 * mean_neg).astype(np.float64)
            x_max  = np.maximum(x_pos, x_neg)
            e_pos  = np.exp(x_pos - x_max)
            e_neg  = np.exp(x_neg - x_max)
            pos_probs_mats[cls] = (e_pos / (e_pos + e_neg)).astype(np.float32)

        # ── Inner grid over (cutoff, window_size, min_in_window, threshold) ──
        for cls in class_names:
            pm     = pos_probs_mats[cls]
            gt_vec = gt_mat[cls]

            for cutoff in cutoff_values:
                for ws, min_in_w in win_pairs:
                    combo_rows = vectorised_f1(
                        pm, valid_mask, n_frames_vec, gt_vec,
                        per_frame_cutoff       = cutoff,
                        window_size            = ws,
                        min_in_window          = min_in_w,
                        threshold_values       = threshold_values,
                        count_threshold_values = count_threshold_values,
                    )
                    for r in combo_rows:
                        r["top_k"]            = top_k
                        r["per_frame_cutoff"] = cutoff
                        r["window_size"]      = ws
                        r["min_in_window"]    = min_in_w
                        all_results[cls].append(r)

                        # Accumulate for global (macro-averaged) tuning.
                        if r["mode"] == "relative":
                            combo_key = (top_k, cutoff, ws, min_in_w, "relative", r["threshold"])
                        else:
                            combo_key = (top_k, cutoff, ws, min_in_w, "absolute", r["count_threshold"])
                        global_f1_accum[combo_key].append(
                            (cls, r["f1"], r["precision"], r["recall"])
                        )

    # Sort by F1 descending, precision as tiebreaker.
    for cls in class_names:
        all_results[cls].sort(key=lambda r: (r["f1"], r["precision"]), reverse=True)

    # ── Build global (macro-averaged) results ─────────────────────────────────
    global_results: List[Dict] = []
    for combo_key, cls_metrics in global_f1_accum.items():
        if len(cls_metrics) != len(class_names):
            continue  # incomplete combo — skip
        top_k, cutoff, ws, min_in_w, mode, thr_or_cnt = combo_key
        f1s   = [m[1] for m in cls_metrics]
        precs = [m[2] for m in cls_metrics]
        recs  = [m[3] for m in cls_metrics]
        entry: Dict = {
            "macro_f1":        float(np.mean(f1s)),
            "macro_precision": float(np.mean(precs)),
            "macro_recall":    float(np.mean(recs)),
            "per_class_f1":    {m[0]: m[1] for m in cls_metrics},
            "top_k":           top_k,
            "per_frame_cutoff": cutoff,
            "window_size":     ws,
            "min_in_window":   min_in_w,
            "mode":            mode,
        }
        if mode == "relative":
            entry["threshold"] = thr_or_cnt
        else:
            entry["count_threshold"] = thr_or_cnt
        global_results.append(entry)
    global_results.sort(key=lambda r: (r["macro_f1"], r["macro_precision"]), reverse=True)
    all_results["_global"] = global_results

    # ── Output ────────────────────────────────────────────────────────────────
    print_global_results(global_results, class_names, args.top_n)
    print_results(all_results, class_names, args.top_n)
    save_outputs(all_results, class_names, args.output, args.best_params, args.global_best_params)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search hyperparameter tuning for classify_clip.py, "
            "optimising F1 on the annotated dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # CLIP model
    parser.add_argument("--model",  default="ViT-B/32",
                        help="CLIP model variant (must match the embedding cache)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device string (cuda / cpu)")

    # Prompt system — mirrors classify_clip.py
    parser.add_argument("--prompt-mode",   default=None,
                        choices=["class-name", "auto", "config"],
                        help="Prompt-bank construction mode (llm not supported here)")
    parser.add_argument("--prompt-config", default="clip_prompts.json", metavar="PATH",
                        help="Prompt config JSON (used when --prompt-mode config)")

    # Dataset / cache
    parser.add_argument("--dataset",        default=DEFAULT_DATASET,
                        help=f"Path to yt_data.json (default: {DEFAULT_DATASET})")
    parser.add_argument("--dataset-root",   default=DEFAULT_DATASET_ROOT,
                        help=f"Root directory of the dataset (default: {DEFAULT_DATASET_ROOT})")
    parser.add_argument("--classes-config", default="classes.json")
    parser.add_argument("--classes",        default=None,
                        help="Comma-separated class names to tune (default: all)")
    parser.add_argument("--embedding-cache", default="clip_embeddings", metavar="DIR",
                        help="Embedding cache directory (must match classify_clip.py run)")
    parser.add_argument("--fps",        type=float, default=4.0,
                        help="Frame sampling rate used when building the cache")
    parser.add_argument("--max-frames", type=int,   default=32, metavar="N",
                        help="Max frames per video used when building the cache")

    # Grid specification
    parser.add_argument(
        "--top-k-values",
        default="3,5,10,20,-1",
        help="Comma-separated top-K values (-1 = all prompts)",
    )
    parser.add_argument(
        "--per-frame-cutoff-values",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated per-frame binary cutoff values",
    )
    parser.add_argument(
        "--window-size-values",
        default="3,5,7,10,15",
        help="Comma-separated sliding-window sizes",
    )
    parser.add_argument(
        "--min-positives-fractions",
        default="0.5,0.6,0.7,0.8,0.9",
        help=(
            "Comma-separated fractions of window_size used to derive "
            "min_positives_in_window (e.g. 0.8 × 10 → 8)"
        ),
    )
    parser.add_argument(
        "--threshold-values",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
        help="Relative active-frame fraction thresholds to try",
    )
    parser.add_argument(
        "--count-threshold-values",
        default="",
        help=(
            "Absolute active-frame count thresholds to try "
            "(empty = disabled; mutually exclusive per evaluation, "
            "but both can be searched in the same run)"
        ),
    )

    # Output
    parser.add_argument("--top-n",       type=int, default=10, metavar="N",
                        help="Number of top results to display per class")
    parser.add_argument("--output",      default="tune_results.json",
                        help="Path for full grid results JSON")
    parser.add_argument("--best-params", default="best_params.json",
                        help="Path for best-params-per-class JSON")
    parser.add_argument("--global-best-params", default="best_params_global.json",
                        help=(
                            "Path for best single set of params maximising macro-averaged F1 "
                            "across all classes (empty string = skip saving)"
                        ))

    return parser.parse_args()


if __name__ == "__main__":
    main()
