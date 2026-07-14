import argparse  # New Import for Command Line Arguments
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASTER = str(REPO_ROOT / "dataset" / "yt_data.json")

# --- Core Utility Function ---


def merge_json_data(master_path, new_data_path, key_field="video_id"):
    """
    Merges new video entries from the new_data_path into the master_path
    JSON file based on a unique key.

    New entries are added only if the key (video_id) is not already present
    in the master file. Existing master entries are never modified.
    """

    print("=" * 60)
    print(f"Starting data merge operation...")
    print(f"Master File: {master_path}")
    print(f"New Data File: {new_data_path}")

    # --- 1. Load Master Data ---
    if os.path.exists(master_path):
        with open(master_path, "r", encoding="utf-8") as f:
            # try:
            master_list = json.load(f)
            print(f"Loaded existing master file with {len(master_list)} entries.")
            # except json.JSONDecodeError:
            #     # If the master file exists but is corrupt, initialize with an empty list
            #     print(
            #         f"❌ Error decoding JSON from master file: {master_path}. Aborting."
            #     )
            #     exit(0)
    else:
        print(
            f"⚠️ Master file not found at {master_path}. Initializing new master list."
        )
        master_list = []

    # --- 2. Create Master Index for Fast Lookup ---
    # Convert the list to a dictionary for O(1) lookup by video_id
    master_index = {
        item.get(key_field): item for item in master_list if item.get(key_field)
    }

    # --- 3. Load New Data ---
    if not os.path.exists(new_data_path):
        print(
            f"❌ Error: Required new data file not found at {new_data_path}. Aborting merge."
        )
        # We don't exit the entire program here, just the function
        return

    with open(new_data_path, "r", encoding="utf-8") as f:
        try:
            new_data_list = json.load(f)
            print(f"Loaded new data file with {len(new_data_list)} entries.")
        except json.JSONDecodeError:
            print(
                f"❌ Error decoding JSON from new data file: {new_data_path}. Aborting merge."
            )
            return

    # --- 4. Perform the Merge ---

    videos_added = 0
    videos_skipped = 0

    for new_item in new_data_list:
        video_id = new_item.get(key_field)

        if not video_id:
            print(f"⚠️ Warning: Skipping item with missing '{key_field}'.")
            continue

        # Check if the video ID is already in the master index
        if video_id in master_index:
            # Rule: DO NOT ALTER existing entries
            videos_skipped += 1
        else:
            # Rule: ADD the new entry
            master_list.append(new_item)
            master_index[video_id] = (
                new_item  # Update the index for any subsequent checks
            )
            videos_added += 1

    # --- 5. Save the Updated Master File ---

    if videos_added > 0 or not os.path.exists(master_path):
        # Only save if new data was added OR if we are creating the file for the first time
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_list, f, ensure_ascii=False, indent=4)

        print("\n✅ Merge successful!")
        print(f"   -> Total new unique videos added: {videos_added}")
        print(f"   -> Total duplicates skipped: {videos_skipped}")
        print(f"   -> Final total videos in master file: {len(master_list)}")
    else:
        print("\nℹ️ No new unique videos found. Master file remains unchanged.")
        print(f"   -> Total duplicates skipped: {videos_skipped}")

    print("=" * 60)


if __name__ == "__main__":

    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Merge new JSON data into a master JSON file, skipping existing entries.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required Positional Argument for New Data
    parser.add_argument(
        "new_data_file",
        type=str,
        help="REQUIRED: The path to the JSON file containing the new data to be merged (e.g., trending_data_shorts.json).",
    )

    # Optional Argument for Master Data
    parser.add_argument(
        "--master",
        type=str,
        default=DEFAULT_MASTER,
        help=f"OPTIONAL: The path to the master JSON file. Defaults to {DEFAULT_MASTER}.",
    )

    args = parser.parse_args()

    # 2. Call the merge function with parsed arguments
    merge_json_data(args.master, args.new_data_file)
