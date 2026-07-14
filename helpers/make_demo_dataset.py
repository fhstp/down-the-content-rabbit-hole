#!/usr/bin/env python3
"""
Generate a small synthetic demo dataset so the full pipeline can be tried
without a YouTube API key.

Creates (by default under <repo_root>/dataset/):
  yt_data.json               master dataset: 8 videos with multi-annotator entries
  videos/demo*.mp4           10-second ffmpeg test-pattern clips (requires ffmpeg on PATH)
  video_data_Animal_0.json   scraper-style per-query file (1 duplicate + 2 new
                             videos) for trying annotation/merge_data.py

The clips are colour test patterns, not real content, so CLIP will correctly
classify them all as negative — the demo exercises the pipeline mechanics
(annotation app, merging, classification, caching, tuning, evaluation), not
classifier quality. Swap in real .mp4 files under videos/ to see real scores.

Usage:
    python helpers/make_demo_dataset.py            # writes to <repo_root>/dataset
    python helpers/make_demo_dataset.py --output /tmp/demo
    python helpers/make_demo_dataset.py --force    # overwrite existing yt_data.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "dataset"

ANNOTATORS = ["alice", "bob", "carol"]

# (video_id, source_class, labels per annotator)
# Mix of full agreement, disagreement, "no label fits" and single annotations,
# so evaluation, agreement statistics and the annotation app all have work to do.
SPEC = [
    ("demo01", "Animal", [["Animal"], ["Animal"], ["Animal"]]),
    ("demo02", "Animal", [["Animal"], ["Animal"]]),
    ("demo03", "Animal", [["Animal"], ["Car"]]),
    ("demo04", "Animal", [[], []]),
    ("demo05", "Animal", [["Animal"]]),
    ("demo06", "Car", [["Car"], ["Car"], ["Car"]]),
    ("demo07", "Car", [["Car"], ["Car"]]),
    ("demo08", "Car", [["Car"], []]),
]

# Extra videos only present in the per-query scraper file (plus one duplicate),
# for demonstrating annotation/merge_data.py. They also get video files so they
# can be annotated after merging.
MERGE_SPEC = [("demo01", "Animal"), ("demo09", "Animal"), ("demo10", "Animal")]


def video_entry(video_id: str, source_class: str, index: int) -> dict:
    return {
        "rank": index + 1,
        "video_id": video_id,
        "source_class": source_class,
        "source_query": f'"demo {source_class.lower()} query"',
        "title": f"Demo {source_class} video {index + 1}",
        "published_at": f"2026-02-{(index % 27) + 1:02d}T12:00:00Z",
        "channel_title": f"demo_channel_{index % 3}",
        "category_id": "15" if source_class == "Animal" else "2",
        "view_count": str(1000 * (index + 1)),
        "like_count": str(50 * (index + 1)),
        "duration": f"PT{20 + 3 * index}S",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "local_filepath": f"videos/{video_id}.mp4",
    }


def render_test_clip(target: Path, source_class: str) -> None:
    """Render a 10s portrait test-pattern clip with ffmpeg."""
    pattern = "testsrc" if source_class == "Animal" else "smptebars"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"{pattern}=duration=10:size=240x426:rate=12",
            "-pix_fmt", "yuv420p", str(target),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output dataset directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing yt_data.json in the output directory",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("error: ffmpeg not found on PATH — required to render demo clips.")

    out = Path(args.output)
    master_path = out / "yt_data.json"
    if master_path.exists() and not args.force:
        sys.exit(
            f"error: {master_path} already exists — refusing to overwrite a "
            "dataset. Pass --force to replace it."
        )

    videos_dir = out / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    master = []
    for i, (video_id, source_class, annotator_labels) in enumerate(SPEC):
        entry = video_entry(video_id, source_class, i)
        entry["target_annotations"] = 3 if len(annotator_labels) >= 3 else 2
        entry["annotation_entries"] = [
            {
                "annotations": labels,
                "annotated_at": f"2026-03-{j + 1:02d}T10:00:00",
                "annotated_by": ANNOTATORS[j],
            }
            for j, labels in enumerate(annotator_labels)
        ]
        master.append(entry)
        render_test_clip(videos_dir / f"{video_id}.mp4", source_class)

    master_path.write_text(json.dumps(master, ensure_ascii=False, indent=4))

    per_query = []
    for i, (video_id, source_class) in enumerate(MERGE_SPEC):
        per_query.append(video_entry(video_id, source_class, i))
        clip = videos_dir / f"{video_id}.mp4"
        if not clip.exists():
            render_test_clip(clip, source_class)
    per_query_path = out / "video_data_Animal_0.json"
    per_query_path.write_text(json.dumps(per_query, ensure_ascii=False, indent=4))

    print(f"Demo dataset written to {out}/")
    print(f"  {master_path.name}: {len(master)} annotated videos")
    print(f"  {per_query_path.name}: {len(MERGE_SPEC)} scraped videos "
          "(1 duplicate) — try annotation/merge_data.py on it")
    print(f"  videos/: {len(list(videos_dir.glob('*.mp4')))} test clips")


if __name__ == "__main__":
    main()
