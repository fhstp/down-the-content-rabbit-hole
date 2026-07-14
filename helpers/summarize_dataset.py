#!/usr/bin/env python3
"""
Dataset summary statistics for the YouTube Shorts dataset.

Produces both plain-text ASCII tables and LaTeX tabular blocks suitable
for inclusion in a paper.

Usage:
    python scrape/summarize_dataset.py
"""

import json
import re
import math
from collections import Counter, defaultdict
from statistics import mean, median, stdev
from pathlib import Path
from datetime import datetime, timezone

# ── Configuration ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "dataset" / "yt_data.json"


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_duration_seconds(dur: str) -> int:
    """Convert an ISO 8601 duration string (e.g. 'PT1M30S') to total seconds."""
    m = re.match(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", dur)
    if not m:
        return 0
    days, hours, minutes, seconds = (int(x) if x else 0 for x in m.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def stats_of(values: list) -> dict:
    """Return descriptive statistics for a non-empty list of numbers."""
    if not values:
        return None
    return {
        "mean":   mean(values),
        "median": median(values),
        "min":    min(values),
        "max":    max(values),
        "stdev":  stdev(values) if len(values) > 1 else 0.0,
    }


# ── Printing helpers ───────────────────────────────────────────────────────────
def print_section(title: str) -> None:
    bar = "═" * max(60, len(title) + 4)
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _strip_latex(s: str) -> str:
    """Convert simple LaTeX markup to plain-text equivalents."""
    s = str(s)
    s = s.replace("\\%",    "%")
    s = s.replace("\\geq",  "≥")
    s = s.replace("\\kappa","κ")
    s = s.replace("\\alpha","α")
    s = s.replace("\\beta", "β")
    # Strip remaining $ delimiters and \cmd{...} wrappers
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+",            "",    s)
    return s


def _col_widths(headers: list, rows: list) -> list:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths


def print_plain(headers: list, rows: list) -> None:
    p_headers = [_strip_latex(h) for h in headers]
    p_rows    = [[_strip_latex(c) for c in row] for row in rows]
    widths    = _col_widths(p_headers, p_rows)
    sep       = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells):
        return "|" + "|".join(
            f" {str(c).ljust(widths[i])} " for i, c in enumerate(cells)
        ) + "|"

    print(sep)
    print(fmt_row(p_headers))
    print(sep)
    for row in p_rows:
        print(fmt_row(row))
    print(sep)


def print_latex(headers: list, rows: list, caption: str = "", label: str = "") -> None:
    ncols = len(headers)
    col_spec = "l" + "r" * (ncols - 1)
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print(f"  \\caption{{{caption}}}")
    print(f"  \\label{{{label}}}")
    print(f"  \\begin{{tabular}}{{{col_spec}}}")
    print("    \\toprule")
    print("    " + " & ".join(str(h) for h in headers) + " \\\\")
    print("    \\midrule")
    for row in rows[:-1]:
        print("    " + " & ".join(str(c) for c in row) + " \\\\")
    # Last row (often a total) separated by a midrule
    print("    \\midrule")
    print("    " + " & ".join(str(c) for c in rows[-1]) + " \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")


def both(headers: list, rows: list, caption: str = "", label: str = "") -> None:
    print("\n[Plain text]")
    print_plain(headers, rows)
    print("\n[LaTeX]")
    print_latex(headers, rows, caption=caption, label=label)


# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    _raw = json.load(f)

# Exclude videos that have never been seen by any annotator (empty annotation_entries).
# Videos where annotators looked but assigned no label are kept.
data = [v for v in _raw if v["annotation_entries"]]

CLASSES = sorted(set(v["source_class"] for v in data))


# ══════════════════════════════════════════════════════════════════════════════
#  1. Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
print_section("1. Dataset Overview")

class_counts = Counter(v["source_class"] for v in data)

dates = [
    datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
    for v in data
]
date_min = min(dates).strftime("%Y-%m-%d")
date_max = max(dates).strftime("%Y-%m-%d")

print(f"\n  Total videos : {len(data):,}")
print(f"  Date range   : {date_min} – {date_max}")
print(f"  Classes      : {len(CLASSES)}")

headers = ["Class", "Videos"]
rows = [(cls, class_counts[cls]) for cls in CLASSES]
rows.append(("Total", len(data)))

both(
    headers, rows,
    caption="Number of videos per source class.",
    label="tab:class_counts",
)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Video Metadata Statistics
# ══════════════════════════════════════════════════════════════════════════════
print_section("2. Video Metadata Statistics")


def build_series(cls_filter=None):
    vids = [v for v in data if cls_filter is None or v["source_class"] == cls_filter]
    durations = [parse_duration_seconds(v["duration"]) for v in vids]
    views     = [int(v["view_count"]) for v in vids]
    likes     = [int(v["like_count"]) for v in vids]
    return durations, views, likes


# 2a. Duration
print("\n--- Duration (seconds) ---")
headers = ["Class", "Mean", "Median", "Min", "Max", "Std Dev"]
rows = []
for cls in CLASSES:
    durs, _, _ = build_series(cls)
    s = stats_of(durs)
    rows.append([cls,
                 f"{s['mean']:.1f}", f"{s['median']:.1f}",
                 s["min"], s["max"], f"{s['stdev']:.1f}"])
durs_all, _, _ = build_series()
s = stats_of(durs_all)
rows.append(["Overall",
             f"{s['mean']:.1f}", f"{s['median']:.1f}",
             s["min"], s["max"], f"{s['stdev']:.1f}"])

both(
    headers, rows,
    caption="Duration statistics (seconds) per source class.",
    label="tab:duration_stats",
)

# 2b. View count
print("\n--- View Count ---")
headers = ["Class", "Mean", "Median", "Min", "Max"]
rows = []
for cls in CLASSES:
    _, views, _ = build_series(cls)
    s = stats_of(views)
    rows.append([cls,
                 f"{s['mean']:,.0f}", f"{s['median']:,.0f}",
                 f"{s['min']:,}", f"{s['max']:,}"])
_, views_all, _ = build_series()
s = stats_of(views_all)
rows.append(["Overall",
             f"{s['mean']:,.0f}", f"{s['median']:,.0f}",
             f"{s['min']:,}", f"{s['max']:,}"])

both(
    headers, rows,
    caption="View count statistics per source class.",
    label="tab:view_stats",
)

# 2c. Like count
print("\n--- Like Count ---")
headers = ["Class", "Mean", "Median", "Min", "Max"]
rows = []
for cls in CLASSES:
    _, _, likes = build_series(cls)
    s = stats_of(likes)
    rows.append([cls,
                 f"{s['mean']:,.1f}", f"{s['median']:,.0f}",
                 f"{s['min']:,}", f"{s['max']:,}"])
_, _, likes_all = build_series()
s = stats_of(likes_all)
rows.append(["Overall",
             f"{s['mean']:,.1f}", f"{s['median']:,.0f}",
             f"{s['min']:,}", f"{s['max']:,}"])

both(
    headers, rows,
    caption="Like count statistics per source class.",
    label="tab:like_stats",
)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Annotation Coverage
# ══════════════════════════════════════════════════════════════════════════════
print_section("3. Annotation Coverage")


def annotation_count(v) -> int:
    return len(v["annotation_entries"])


coverage     = {cls: Counter(annotation_count(v) for v in data if v["source_class"] == cls)
                for cls in CLASSES}
coverage_all = Counter(annotation_count(v) for v in data)

headers = ["Class", "Videos", "0 ann.", "1 ann.", "2 ann.", "3 ann.", "Complete (%)"]
rows = []
for cls in CLASSES:
    n = class_counts[cls]
    c = coverage[cls]
    pct = 100.0 * c[3] / n if n else 0.0
    rows.append([cls, n, c[0], c[1], c[2], c[3], f"{pct:.1f}\\%"])

n_all   = len(data)
pct_all = 100.0 * coverage_all[3] / n_all
rows.append(["Total", n_all,
             coverage_all[0], coverage_all[1], coverage_all[2], coverage_all[3],
             f"{pct_all:.1f}\\%"])

both(
    headers, rows,
    caption=(
        "Annotation coverage per source class. "
        "\\emph{Complete (\\%)} is the percentage of videos that have "
        "reached the target of 3 independent annotations."
    ),
    label="tab:annotation_coverage",
)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Label Distribution (Confusion Matrix)
# ══════════════════════════════════════════════════════════════════════════════
print_section("4. Label Distribution — Source Class vs. Assigned Label")

# Build confusion matrix counting DISTINCT VIDEOS, not annotation events.
# For each video we collect the set of unique labels assigned by any annotator,
# then increment each (source_class, label) cell by 1 once per unique label.
# A video with two annotators both saying "Animal" counts as 1, not 2.
# A video where annotators disagreed (e.g. "Animal" and "Car") contributes 1
# to both columns, so row cells can sum to more than the video count.
confusion: dict[str, Counter] = defaultdict(Counter)
for v in data:
    cls            = v["source_class"]
    labels_for_vid: set[str] = set()
    for ae in v["annotation_entries"]:
        if ae["annotations"]:
            labels_for_vid.update(ae["annotations"])
        else:
            labels_for_vid.add("(none)")
    for lbl in labels_for_vid:
        confusion[cls][lbl] += 1

all_labels_used = sorted(
    set(lbl for c in confusion.values() for lbl in c),
    key=lambda x: (x == "(none)", x),       # put "(none)" last
)

headers = ["Source class"] + all_labels_used + ["Videos"]
rows = []
for cls in CLASSES:
    row = [cls]
    for lbl in all_labels_used:
        row.append(confusion[cls].get(lbl, 0))
    # Total = number of distinct videos in this class (not sum of cells)
    row.append(class_counts[cls])
    rows.append(row)

# Column totals row
col_totals = (
    ["Total"]
    + [sum(confusion[cls].get(lbl, 0) for cls in CLASSES) for lbl in all_labels_used]
    + [len(data)]
)
rows.append(col_totals)

both(
    headers, rows,
    caption=(
        "Confusion matrix: number of \\emph{videos} (not annotation events) for which "
        "at least one annotator assigned each label, broken down by source class. "
        "A video where annotators disagreed is counted in each label they used, so "
        "row cells can sum to more than \\emph{Videos}. "
        "\\emph{(none)} indicates at least one annotator found no matching label."
    ),
    label="tab:confusion_matrix",
)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Inter-Annotator Agreement
# ══════════════════════════════════════════════════════════════════════════════
print_section("5. Inter-Annotator Agreement")


def get_label(ae: dict) -> str:
    """Return the single label for an annotation entry, or '' if none."""
    return ae["annotations"][0] if ae["annotations"] else ""


# Collect all pairwise (label_A, label_B) for every video with >= 2 annotations
all_pairs: list[tuple[str, str]] = []
for v in data:
    entries = v["annotation_entries"]
    if len(entries) < 2:
        continue
    labels = [get_label(ae) for ae in entries]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            all_pairs.append((labels[i], labels[j]))

n_pairs = len(all_pairs)
n_agree = sum(1 for a, b in all_pairs if a == b)
po = n_agree / n_pairs if n_pairs else 0.0

# Expected agreement from pooled marginal distribution (symmetric kappa)
all_labels_flat = [lbl for pair in all_pairs for lbl in pair]
label_freq      = Counter(all_labels_flat)
n_flat          = len(all_labels_flat)
pe              = sum((cnt / n_flat) ** 2 for cnt in label_freq.values())
kappa           = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0.0 else 0.0

n_videos_multi = sum(1 for v in data if len(v["annotation_entries"]) >= 2)

print(f"\n  Videos with ≥ 2 annotations : {n_videos_multi:,}")
print(f"  Annotation pairs evaluated  : {n_pairs:,}")
print(f"  Pairs in agreement          : {n_agree:,}")
print(f"  Observed agreement (P_o)    : {po:.4f}  ({100*po:.1f}%)")
print(f"  Expected agreement (P_e)    : {pe:.4f}")
print(f"  Cohen's kappa (κ)           : {kappa:.4f}")

headers = ["Metric", "Value"]
rows = [
    ["Videos with \\geq 2 annotations", f"{n_videos_multi:,}"],
    ["Annotation pairs evaluated",       f"{n_pairs:,}"],
    ["Pairs in agreement",               f"{n_agree:,}"],
    ["Observed agreement ($P_o$)",       f"{po:.4f}"],
    ["Expected agreement ($P_e$)",       f"{pe:.4f}"],
    ["Cohen's kappa ($\\kappa$)",        f"{kappa:.4f}"],
]

both(
    headers, rows,
    caption="Inter-annotator agreement statistics computed over all pairwise comparisons.",
    label="tab:iaa",
)

# ── Pre-compute per-class and per-label stats for the combined table ───────────

# Source class → (pairs, agree, pct)
class_stats: dict = {}
for cls in CLASSES:
    cp = []
    for v in data:
        if v["source_class"] != cls:
            continue
        entries = v["annotation_entries"]
        if len(entries) < 2:
            continue
        labels = [get_label(ae) for ae in entries]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                cp.append((labels[i], labels[j]))
    ag = sum(1 for a, b in cp if a == b)
    class_stats[cls] = (len(cp), ag, 100.0 * ag / len(cp)) if cp else (0, 0, float("nan"))

# Annotation label → (pairs, agree, pct)  ('' = no label assigned)
unique_labels = sorted(
    set(lbl for pair in all_pairs for lbl in pair),
    key=lambda x: (x == "", x),   # put (none) last
)
label_stats: dict = {}
for lbl in unique_labels:
    inv = [(a, b) for a, b in all_pairs if lbl in (a, b)]
    ag  = sum(1 for a, b in inv if a == lbl and b == lbl)
    label_stats[lbl] = (len(inv), ag, 100.0 * ag / len(inv)) if inv else (0, 0, float("nan"))

# Map each source class to its corresponding annotation label name.
# Annotation labels match source-class names; override entries here if a
# class is annotated under a different label.
CLASS_TO_LABEL = {cls: cls for cls in CLASSES}
# Row display name (uses annotation-label wording)
ROW_DISPLAY = {cls: cls for cls in CLASSES}


def _pct(p) -> str:
    return f"{p:.1f}\\%" if not math.isnan(p) else "—"


# Build the combined data rows
comb_rows = []
for cls in CLASSES:
    lbl           = CLASS_TO_LABEL[cls]
    cp, ca, cpct  = class_stats[cls]
    lp, la, lpct  = label_stats.get(lbl, (0, 0, float("nan")))
    comb_rows.append([ROW_DISPLAY[cls], cp, ca, _pct(cpct), lp, la, _pct(lpct)])

# (none) row — has no source-class counterpart
lp, la, lpct = label_stats.get("", (0, 0, float("nan")))
comb_rows.append(["(none)", "—", "—", "—", lp, la, _pct(lpct)])

# Total / Overall row (same pair pool viewed from two angles)
comb_rows.append(["Total / Overall",
                  n_pairs, n_agree, _pct(100 * po),
                  n_pairs, n_agree, _pct(100 * po)])

col_h = ["Category", "Pairs", "Agree", "Agr.%", "Pairs", "Agree", "Agr.%"]

# ── Plain text ─────────────────────────────────────────────────────────────────
print("\n--- Agreement by source class and annotation label (side-by-side) ---")
print("\n[Plain text]")

p_col_h   = [_strip_latex(h) for h in col_h]
p_rows    = [[_strip_latex(str(c)) for c in row] for row in comb_rows]
widths    = _col_widths(p_col_h, p_rows)
sep       = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

def _fr(cells):
    return "|" + "|".join(f" {str(c).ljust(widths[i])} " for i, c in enumerate(cells)) + "|"

# Group-header line: centred over their respective column spans
# Each cell occupies (width + 2) chars; separators add 1 char between cells.
g1_w = sum(widths[1:4]) + 2 * 3 + 2   # 3 cells: padding + 2 inner separators
g2_w = sum(widths[4:7]) + 2 * 3 + 2
indent = widths[0] + 2 + 2             # category cell width + padding + leading "|"
print(" " * indent + "By source class".center(g1_w) + " " + "By annotation label".center(g2_w))
print(sep)
print(_fr(p_col_h))
print(sep)
for row in p_rows[:-1]:
    print(_fr(row))
print(sep)
print(_fr(p_rows[-1]))   # Total / Overall with preceding separator
print(sep)

# ── LaTeX ──────────────────────────────────────────────────────────────────────
print("\n[LaTeX]")
caption = (
    "Pairwise inter-annotator agreement broken down by source class (left) and "
    "annotation label (right). "
    "For source class, \\emph{Pairs} counts all pairwise comparisons from videos "
    "in that class. "
    "For annotation label, \\emph{Pairs} counts pairs where at least one annotator "
    "assigned that label; a disagreeing pair may appear in multiple rows, so the "
    "column does not sum to \\emph{Total}. "
    "\\emph{(none)} indicates at least one annotator found no matching label."
)
print("\\begin{table}[htbp]")
print("  \\centering")
print(f"  \\caption{{{caption}}}")
print("  \\label{tab:iaa_combined}")
print("  \\begin{tabular}{lrrrrrr}")
print("    \\toprule")
print("    & \\multicolumn{3}{c}{By source class}"
      " & \\multicolumn{3}{c}{By annotation label} \\\\")
print("    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
print("    Category & Pairs & Agree & Agr.\\%"
      " & Pairs & Agree & Agr.\\% \\\\")
print("    \\midrule")
for row in comb_rows[:-1]:
    print("    " + " & ".join(str(c) for c in row) + " \\\\")
print("    \\midrule")
print("    " + " & ".join(str(c) for c in comb_rows[-1]) + " \\\\")
print("    \\bottomrule")
print("  \\end{tabular}")
print("\\end{table}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. Annotator Statistics
# ══════════════════════════════════════════════════════════════════════════════
print_section("6. Annotator Statistics")

annotator_counts: Counter = Counter()
for v in data:
    for ae in v["annotation_entries"]:
        annotator_counts[ae["annotated_by"]] += 1

total_ann = sum(annotator_counts.values())
print(f"\n  Unique annotators : {len(annotator_counts)}")
print(f"  Total annotations : {total_ann:,}")

headers = ["Annotator", "Annotations", "Share (%)"]
rows = []
for ann, cnt in annotator_counts.most_common():
    rows.append([ann, cnt, f"{100*cnt/total_ann:.1f}\\%"])
rows.append(["Total", total_ann, "100.0\\%"])

both(
    headers, rows,
    caption="Number of annotations contributed by each annotator.",
    label="tab:annotators",
)

print("\n" + "─" * 60)
print("  Done.")
print("─" * 60 + "\n")
