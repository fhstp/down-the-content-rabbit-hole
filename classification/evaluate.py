"""
Evaluation script for classification results produced by classify_clip.py.

Usage:
    python evaluate.py --clip-results classification_results_clip.json

Defaults --data / --dataset-root to <repo_root>/dataset.
Omit --model / --label to evaluate every model × class combo found in the results file(s).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = str(REPO_ROOT / "dataset" / "yt_data.json")
DEFAULT_DATASET_ROOT = str(REPO_ROOT / "dataset")


# ── Frame / timestamp helpers ─────────────────────────────────────────────────

def frames_to_ranges(frame_numbers: List[int], frame_timestamps_s: List[float]) -> List[str]:
    """Convert a list of 1-based frame numbers to human-readable time-range strings.

    Consecutive frames are collapsed into a single span.  Each range is formatted
    as "start_s-end_s" where the end is the timestamp of the last frame in the run.

    Example:
        frames [3, 4, 5, 12, 13], 4fps sampling
        → ["0.50s-1.00s", "2.75s-3.00s"]
    """
    if not frame_numbers or not frame_timestamps_s:
        return []

    # Clamp to valid range (model may hallucinate out-of-bounds frame numbers).
    n = len(frame_timestamps_s)
    valid = sorted(f for f in frame_numbers if 1 <= f <= n)
    if not valid:
        return []

    ranges: List[str] = []
    run_start = run_end = valid[0]
    for f in valid[1:]:
        if f == run_end + 1:
            run_end = f
        else:
            ranges.append(
                f"{frame_timestamps_s[run_start - 1]:.2f}s"
                f"-{frame_timestamps_s[run_end - 1]:.2f}s"
            )
            run_start = run_end = f
    ranges.append(
        f"{frame_timestamps_s[run_start - 1]:.2f}s"
        f"-{frame_timestamps_s[run_end - 1]:.2f}s"
    )
    return ranges


def resolve_frames(result: dict, frame_map_entry: Optional[dict]) -> Optional[str]:
    """Return a compact timestamp string for a single video result, or None."""
    frames = result.get("frames") if isinstance(result, dict) else None
    if not frames or not frame_map_entry:
        return None
    ranges = frames_to_ranges(frames, frame_map_entry.get("frame_timestamps_s", []))
    return ", ".join(ranges) if ranges else None


# ── Data loading ──────────────────────────────────────────────────────────────

def derive_consensus_labels(entries: List[Dict]) -> List[str]:
    """Majority-vote consensus over per-annotator annotation entries.

    A label is part of the consensus if strictly more than half of the
    annotators assigned it.  With a single annotator, their labels are the
    consensus; with two annotators, both must agree.
    """
    counts: Dict[str, int] = {}
    for e in entries:
        for lbl in e.get("annotations", []):
            counts[lbl] = counts.get(lbl, 0) + 1
    quorum = len(entries) / 2
    return sorted(lbl for lbl, c in counts.items() if c > quorum)


def load_annotated_data(json_path: str, dataset_root: str) -> List[Dict]:
    """Load videos that have at least one annotation entry.

    Ground-truth labels are derived per video as the majority vote across
    its annotation_entries and stored in entry["consensus_labels"].
    """
    with open(json_path) as f:
        data = json.load(f)
    annotated = [e for e in data if e.get("annotation_entries")]
    for entry in annotated:
        entry["consensus_labels"] = derive_consensus_labels(entry["annotation_entries"])
        lp = entry.get("local_filepath")
        if lp:
            entry["absolute_path"] = os.path.join(dataset_root, lp)
        else:
            entry["absolute_path"] = os.path.join(
                dataset_root, "videos", f"{entry['video_id']}.mp4"
            )
    return annotated


# ── Per-combo evaluation ──────────────────────────────────────────────────────

def evaluate_combo(
    results_for_class: Dict,   # {video_id: {"classification": bool, ...} | str | {"error":...}}
    data: List[Dict],
    label: str,
) -> Dict:
    tp = fp = tn = fn = 0
    tp_ids: List[str] = []
    fp_ids: List[str] = []
    tn_ids: List[str] = []
    fn_ids: List[str] = []
    skipped = 0

    for entry in data:
        video_id = entry.get("video_id", "")

        has_label = label in entry.get("consensus_labels", [])

        result = results_for_class.get(video_id)
        if result is None:
            skipped += 1
            continue

        # Gracefully handle error / unparsed results.
        if isinstance(result, str) or (isinstance(result, dict) and "error" in result):
            skipped += 1
            continue

        # Tolerate legacy list-wrapped results.
        if isinstance(result, list):
            result = result[0]

        predicted = bool(result.get("classification", False))

        if has_label and predicted:
            tp += 1
            tp_ids.append(video_id)
        elif has_label and not predicted:
            fn += 1
            fn_ids.append(video_id)
        elif not has_label and predicted:
            fp += 1
            fp_ids.append(video_id)
        else:
            tn += 1
            tn_ids.append(video_id)

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total":     total,
        "skipped":   skipped,
        "positives": tp + fn,
        "negatives": tn + fp,
        "tp_ids": tp_ids,
        "fp_ids": fp_ids,
        "tn_ids": tn_ids,
        "fn_ids": fn_ids,
        # raw results keyed by video_id for frame/timestamp resolution in print_combo
        "_results": results_for_class,
    }


# ── Timing helpers ────────────────────────────────────────────────────────────

def compute_timing_stats(timing: Dict, label: Optional[str] = None) -> Optional[Dict]:
    """Compute average timing stats from the _timing dict for a model.

    If *label* is given, reports the average per-class time for that specific
    class.  Always reports average encode, total, and per-class-call times.
    Returns None if no timing data is available.
    """
    if not timing:
        return None

    encode_times: List[float] = []
    total_times: List[float] = []
    class_times: List[float] = []  # per-class-call durations for *label*

    for vid, t in timing.items():
        if not isinstance(t, dict):
            continue
        encode_times.append(t.get("encode_s", 0.0))
        total_times.append(t.get("total_s", 0.0))
        if label:
            per_class = t.get("per_class_s", {})
            if label in per_class:
                class_times.append(per_class[label])

    if not total_times:
        return None

    n = len(total_times)
    stats: Dict = {
        "videos": n,
        "avg_encode_s": sum(encode_times) / n,
        "avg_total_s": sum(total_times) / n,
    }
    if class_times:
        stats["avg_class_s"] = sum(class_times) / len(class_times)
    return stats

def print_combo(
    metrics: Dict,
    model: str,
    label: str,
    timing_stats: Optional[Dict] = None,
    frame_map: Optional[Dict] = None,
    show_ids: bool = False,
) -> None:
    print(f"\n{'=' * 62}")
    print(f"  Model: {model}  |  Label: {label}")
    print(f"{'=' * 62}")
    print(
        f"  Evaluated: {metrics['total']}  "
        f"(+: {metrics['positives']}, -: {metrics['negatives']}, "
        f"skipped: {metrics['skipped']})"
    )
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {metrics['tp']:4d}    FP: {metrics['fp']:4d}")
    print(f"    FN: {metrics['fn']:4d}    TN: {metrics['tn']:4d}")
    print(f"\n  Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']  * 100:.1f}%)")
    print(f"    Precision: {metrics['precision']:.4f}  ({metrics['precision'] * 100:.1f}%)")
    print(f"    Recall:    {metrics['recall']:.4f}  ({metrics['recall']    * 100:.1f}%)")
    print(f"    F1:        {metrics['f1']:.4f}")
    if show_ids:
        if metrics["fp_ids"]:
            print(f"\n  False Positives ({len(metrics['fp_ids'])}):")
            for vid in metrics["fp_ids"]:
                ts = resolve_frames(metrics["_results"].get(vid), (frame_map or {}).get(vid))
                suffix = f"  [{ts}]" if ts else ""
                print(f"    {vid}{suffix}")
        if metrics["fn_ids"]:
            print(f"\n  False Negatives ({len(metrics['fn_ids'])}):")
            for vid in metrics["fn_ids"]:
                ts = resolve_frames(metrics["_results"].get(vid), (frame_map or {}).get(vid))
                suffix = f"  [{ts}]" if ts else ""
                print(f"    {vid}{suffix}")
    if timing_stats:
        print(f"\n  Timing (over {timing_stats['videos']} videos):")
        print(f"    Avg encode:        {timing_stats['avg_encode_s']:>7.2f}s")
        if "avg_class_s" in timing_stats:
            print(f"    Avg classify ({label}): {timing_stats['avg_class_s']:>7.2f}s")
        print(f"    Avg total/video:   {timing_stats['avg_total_s']:>7.2f}s")


def print_summary(all_metrics: List[Tuple[str, str, Dict, Optional[Dict]]]) -> None:
    print(f"\n{'=' * 96}")
    print(f"  SUMMARY")
    print(f"{'=' * 96}")
    header = (
        f"  {'Model':<14} {'Label':<22} "
        f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
        f"{'Avg/vid':>8}"
    )
    print(header)
    print(f"  {'-' * 92}")
    for model, label, m, ts in all_metrics:
        avg_s = f"{ts['avg_total_s']:.2f}s" if ts else "-"
        print(
            f"  {model:<14} {label:<22} "
            f"{m['accuracy']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}  "
            f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}  "
            f"{avg_s:>8}"
        )
    print(f"{'=' * 96}")


# ── LaTeX output ─────────────────────────────────────────────────────────────

def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in a plain text string."""
    # Order matters: backslash must come first.
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def _latex_label(s: str) -> str:
    """Convert an arbitrary string to a safe LaTeX \\label identifier."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def print_combo_latex(metrics: Dict, model: str, label: str) -> None:
    """Print a LaTeX table for a single (model, label) evaluation result.

    Requires \\usepackage{booktabs} in the document preamble.
    """
    em  = _latex_escape(model)
    el  = _latex_escape(label)
    lbl = _latex_label(f"{model}_{label}")
    n   = metrics["total"]
    print(f"\n% ── LaTeX: {model} / {label} " + "─" * 30)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{rrrr|rrrr}")
    print(r"\toprule")
    print(r"Acc. & Prec. & Rec. & F1 & TP & FP & FN & TN \\")
    print(r"\midrule")
    print(
        f"{metrics['accuracy']  * 100:.1f}\\% & "
        f"{metrics['precision'] * 100:.1f}\\% & "
        f"{metrics['recall']    * 100:.1f}\\% & "
        f"{metrics['f1']:.3f} & "
        f"{metrics['tp']} & {metrics['fp']} & "
        f"{metrics['fn']} & {metrics['tn']} \\\\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\\caption{{Results for {em} / {el} ($N={n}$)}}")
    print(f"\\label{{tab:{lbl}}}")
    print(r"\end{table}")


def print_summary_latex(all_metrics: List[Tuple[str, str, Dict, Optional[Dict]]]) -> None:
    """Print a LaTeX summary table for all (model, label) results.

    Requires \\usepackage{booktabs} in the document preamble.
    """
    # Decide whether any row has timing data — include column only if so.
    has_timing = any(ts is not None for _, _, _, ts in all_metrics)

    col_spec = "ll|rrrr|rrrr" + ("r" if has_timing else "")
    print(f"\n% ── LaTeX: Summary Table " + "─" * 30)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    timing_hdr = r" & Avg/vid" if has_timing else ""
    print(r"Model & Class & Acc. & Prec. & Rec. & F1 & TP & FP & FN & TN" + timing_hdr + r" \\")
    print(r"\midrule")
    prev_model = None
    for model, label, m, ts in all_metrics:
        em = _latex_escape(model)
        el = _latex_escape(label)
        # Insert a \midrule between model groups for readability.
        if prev_model is not None and model != prev_model:
            print(r"\midrule")
        prev_model = model
        timing_col = ""
        if has_timing:
            avg_s = f"{ts['avg_total_s']:.2f}s" if ts else "---"
            timing_col = f" & {avg_s}"
        print(
            f"{em} & {el} & "
            f"{m['accuracy']  * 100:.1f}\\% & "
            f"{m['precision'] * 100:.1f}\\% & "
            f"{m['recall']    * 100:.1f}\\% & "
            f"{m['f1']:.3f} & "
            f"{m['tp']} & {m['fp']} & "
            f"{m['fn']} & {m['tn']}"
            f"{timing_col} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Classification results summary}")
    print(r"\label{tab:results_summary}")
    print(r"\end{table}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classification results against human annotations"
    )
    parser.add_argument(
        "--results",
        default="classification_results.json",
        help="Path to classification_results.json (default: classification_results.json)",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATASET,
        help=f"Path to yt_data.json (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"Root directory of the dataset (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate (default: all models in results file)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label/class to evaluate (default: all labels in results file)",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Print video IDs of false positives and false negatives (default: counts only)",
    )
    parser.add_argument(
        "--clip-results",
        default=None,
        metavar="PATH",
        help=(
            "Path to a CLIP results JSON produced by classify_clip.py "
            "(e.g. classification_results_clip.json).  Models from this file "
            "are merged into the evaluation alongside any --results file."
        ),
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a LaTeX (booktabs) table after each terminal table",
    )
    args = parser.parse_args()

    # ── Load results ──────────────────────────────────────────────────────────
    results: Dict = {}
    if args.results and os.path.exists(args.results) and os.path.getsize(args.results) > 0:
        with open(args.results) as f:
            results = json.load(f)
    elif args.results:
        print(f"Note: --results file not found or empty: {args.results}")

    # ── Merge CLIP results ────────────────────────────────────────────────────
    if args.clip_results:
        if not os.path.exists(args.clip_results):
            print(f"Warning: --clip-results file not found: {args.clip_results}")
        else:
            with open(args.clip_results) as f:
                clip_results = json.load(f)
            for model_key, model_data in clip_results.items():
                if model_key in results:
                    print(f"Warning: model key '{model_key}' already in results — overwriting with CLIP data.")
                results[model_key] = model_data
            print(f"Loaded CLIP results from {args.clip_results}: {list(clip_results.keys())}")

    data = load_annotated_data(args.data, args.dataset_root)
    print(f"Loaded {len(data)} annotated videos from {args.data}")

    models_to_eval = [args.model] if args.model else list(results.keys())

    all_metrics: List[Tuple[str, str, Dict, Optional[Dict]]] = []
    for model in models_to_eval:
        if model not in results:
            print(f"Warning: model '{model}' not found in results")
            continue
        timing = results[model].get("_timing", {})
        frame_map = results[model].get("_frame_map", {})
        labels_to_eval = [args.label] if args.label else [
            k for k in results[model] if k not in ("_timing", "_frame_map")
        ]
        for label in labels_to_eval:
            if label not in results[model]:
                print(f"Warning: label '{label}' not found for model '{model}'")
                continue
            metrics = evaluate_combo(results[model][label], data, label)
            ts = compute_timing_stats(timing, label)
            all_metrics.append((model, label, metrics, ts))
            print_combo(metrics, model, label, ts, frame_map or None, show_ids=args.show_ids)
            if args.latex:
                print_combo_latex(metrics, model, label)

    if len(all_metrics) > 1:
        print_summary(all_metrics)
        if args.latex:
            print_summary_latex(all_metrics)


if __name__ == "__main__":
    main()
