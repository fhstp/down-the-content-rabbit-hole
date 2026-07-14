"""
Quick utility to analyze the annotations in yt_data.json
"""

import json
from collections import Counter
from pathlib import Path


def analyze_annotations(json_path: str):
    """Analyze annotation statistics in the dataset."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_videos = len(data)
    # Annotated videos have the 'annotations' field (even if empty)
    annotated_videos = [d for d in data if 'annotations' in d]
    unannotated_videos = total_videos - len(annotated_videos)
    
    # Separate videos with labels vs. videos annotated as "none of the labels fit"
    videos_with_labels = [d for d in annotated_videos if len(d['annotations']) > 0]
    videos_no_labels = [d for d in annotated_videos if len(d['annotations']) == 0]
    
    # Collect all labels
    all_labels = []
    for video in annotated_videos:
        all_labels.extend(video['annotations'])
    
    label_counts = Counter(all_labels)
    
    # Print statistics
    print("=" * 60)
    print("DATASET ANNOTATION STATISTICS")
    print("=" * 60)
    print(f"\nTotal videos in dataset: {total_videos}")
    print(f"Annotated videos: {len(annotated_videos)}")
    print(f"  - With labels: {len(videos_with_labels)}")
    print(f"  - No labels fit: {len(videos_no_labels)}")
    print(f"Unannotated videos: {unannotated_videos}")
    print(f"Annotation coverage: {len(annotated_videos)/total_videos*100:.1f}%")
    
    print(f"\nLabel Distribution:")
    print("-" * 60)
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(annotated_videos) * 100
        print(f"  {label:20s} : {count:3d} videos ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("Available labels for evaluation:")
    print(", ".join(sorted(label_counts.keys())))
    print("=" * 60)
    
    # Check for videos with multiple labels
    multi_label_videos = [v for v in videos_with_labels if len(v['annotations']) > 1]
    if multi_label_videos:
        print(f"\nVideos with multiple labels: {len(multi_label_videos)}")
        print("Example multi-label combinations:")
        label_combinations = Counter([tuple(sorted(v['annotations'])) for v in multi_label_videos])
        for combo, count in sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {', '.join(combo):40s} : {count} video(s)")
    
    print()


if __name__ == '__main__':
    import argparse

    REPO_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_PATH = str(REPO_ROOT / "dataset" / "yt_data.json")

    parser = argparse.ArgumentParser(description='Analyze annotation statistics')
    parser.add_argument('--data-path', type=str,
                        default=DEFAULT_DATA_PATH,
                        help=f'Path to yt_data.json (default: {DEFAULT_DATA_PATH})')

    args = parser.parse_args()

    analyze_annotations(args.data_path)
