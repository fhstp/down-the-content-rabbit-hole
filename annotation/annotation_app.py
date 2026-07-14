"""
Video Annotation Web Application
Flask-based tool for annotating videos from yt_data.json.

Supports:
  - Password-protected login (shared password + annotator name)
  - Multi-annotation: each video can be annotated by multiple people
  - Auto-escalation: once 100 videos reach 2 annotations, target is raised to 3
  - Backward-compatible migration from the old single-annotation schema
"""

import json
import os
import random
import secrets
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

load_dotenv()

app = Flask(__name__)

# Use SECRET_KEY from env; fall back to a per-process random key (sessions won't
# survive restarts, which is acceptable here).
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    _secret_key = secrets.token_hex(32)
    print(
        "WARNING: SECRET_KEY not set in environment. "
        "Sessions will be invalidated on every restart. "
        "Set SECRET_KEY in scrape/.env for persistent sessions."
    )
app.secret_key = _secret_key

# --- Configuration ---
DATA_FILE = str(REPO_ROOT / "dataset" / "yt_data.json")
VIDEO_DIR = str(REPO_ROOT / "dataset" / "videos")
LOCK_TIMEOUT_SECONDS = 3600  # 1 hour

ANNOTATION_PASSWORD = os.environ.get("ANNOTATION_PASSWORD", "")
if not ANNOTATION_PASSWORD:
    print(
        "WARNING: ANNOTATION_PASSWORD is not set. "
        "Set ANNOTATION_PASSWORD in scrape/.env before exposing the app."
    )
    ANNOTATION_PASSWORD = "changeme"

# Default annotation labels (can be reconfigured via the UI at runtime;
# edits are persisted to LABELS_FILE and survive server restarts)
DEFAULT_LABELS = [
    "Animal",
    "Car",
    "Food",
]
LABELS_FILE = str(REPO_ROOT / "dataset" / "annotation_labels.json")


def load_labels():
    """Return persisted labels from LABELS_FILE, or DEFAULT_LABELS."""
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and all(isinstance(l, str) for l in labels):
                return labels
            print(f"WARNING: {LABELS_FILE} is not a list of strings; using defaults.")
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: could not read {LABELS_FILE} ({e}); using defaults.")
    return list(DEFAULT_LABELS)


def save_labels(labels):
    """Persist labels to LABELS_FILE (atomic-ish: write then replace)."""
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    tmp = LABELS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    os.replace(tmp, LABELS_FILE)


current_labels = load_labels()

# Thread lock for concurrent data access
data_lock = threading.Lock()

# In-memory video locks: {video_id: {"locked_at": float, "session_id": str}}
locked_videos = {}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def login_required(f):
    """Decorator: redirect to /login (or return 401 for JSON requests) if not logged in."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if "annotator_name" not in session:
            if request.is_json:
                return jsonify({"success": False, "message": "Not authenticated"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_data():
    """Load video data from JSON file."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    """Save video data to JSON file (atomic-ish: write then replace)."""
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, DATA_FILE)


def migrate_data():
    """
    One-time (idempotent) migration from the old single-annotation schema to the
    new multi-annotation schema.

    Old schema (fields directly on video object):
        "annotations": [...],
        "annotated_at": "...",
        "annotated_by": "...",
        "last_edited_at": "..."   # optional

    New schema:
        "target_annotations": 2,
        "annotation_entries": [
            {
                "annotations": [...],
                "annotated_at": "...",
                "annotated_by": "...",
                "last_edited_at": "..."   # optional, only if previously edited
            }
        ]
    """
    with data_lock:
        data = load_data()
        changed = False

        for video in data:
            # Already migrated — skip
            if "annotation_entries" in video:
                continue

            changed = True

            if "annotations" in video:
                entry = {
                    "annotations": video.pop("annotations"),
                    "annotated_at": video.pop("annotated_at", None),
                    "annotated_by": video.pop("annotated_by", "unknown"),
                }
                if "last_edited_at" in video:
                    entry["last_edited_at"] = video.pop("last_edited_at")
                video["annotation_entries"] = [entry]
            else:
                # Remove any stale partial fields
                for field in ("annotations", "annotated_at", "annotated_by", "last_edited_at"):
                    video.pop(field, None)
                video["annotation_entries"] = []

            video.setdefault("target_annotations", 2)

        if changed:
            save_data(data)
            already_annotated = sum(
                1 for v in data if v.get("annotation_entries")
            )
            print(
                f"  Migration complete. "
                f"{already_annotated} previously-annotated videos wrapped into new schema."
            )
        else:
            print("  Data already in new format — no migration needed.")


def maybe_escalate_targets(data):
    """
    If >= 100 videos have at least 2 annotation entries, escalate all videos
    whose target is still 2 up to 3.  Returns True if any change was made.
    """
    double_annotated = sum(
        1 for v in data if len(v.get("annotation_entries", [])) >= 2
    )
    if double_annotated >= 100:
        changed = False
        for video in data:
            if video.get("target_annotations", 2) == 2:
                video["target_annotations"] = 3
                changed = True
        return changed
    return False


def cleanup_expired_locks():
    """Remove in-memory locks that have expired."""
    now = time.time()
    expired = [
        vid
        for vid, info in locked_videos.items()
        if now - info["locked_at"] > LOCK_TIMEOUT_SECONDS
    ]
    for vid in expired:
        del locked_videos[vid]


def get_next_video(annotator_name, session_id):
    """
    Return a random video that:
      - still needs more annotations (len(entries) < target)
      - has not yet been annotated by *this* annotator
      - is not locked by a *different* session
      - has a local .mp4 file on disk

    Locks the chosen video for this session before returning.
    """
    cleanup_expired_locks()

    with data_lock:
        data = load_data()

        candidates = []
        for video in data:
            video_id = video.get("video_id")
            if not video_id:
                continue

            entries = video.get("annotation_entries", [])
            target = video.get("target_annotations", 2)

            # Enough annotations already
            if len(entries) >= target:
                continue

            # This annotator already contributed
            if any(e.get("annotated_by") == annotator_name for e in entries):
                continue

            # Locked by someone else
            lock = locked_videos.get(video_id)
            if lock and lock["session_id"] != session_id:
                continue

            # Must have a downloadable video file
            if os.path.exists(os.path.join(VIDEO_DIR, f"{video_id}.mp4")):
                candidates.append(video)

        if not candidates:
            return None

        video = random.choice(candidates)

        locked_videos[video["video_id"]] = {
            "locked_at": time.time(),
            "session_id": session_id,
        }

        return video


def unlock_video(video_id, session_id):
    """Release the lock on a video, but only if it belongs to this session."""
    lock = locked_videos.get(video_id)
    if lock and lock["session_id"] == session_id:
        del locked_videos[video_id]


# ---------------------------------------------------------------------------
# Routes — auth
# ---------------------------------------------------------------------------


@app.route("/login", methods=["GET", "POST"])
def login():
    """Shared-password login with annotator name."""
    if "annotator_name" in session:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        annotator_name = request.form.get("annotator_name", "").strip()
        password = request.form.get("password", "")

        if not annotator_name:
            error = "Please enter your name."
        elif password != ANNOTATION_PASSWORD:
            error = "Incorrect password."
        else:
            session["annotator_name"] = annotator_name
            session["session_id"] = secrets.token_hex(16)
            return redirect(url_for("index"))

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Log out, releasing any locks held by this session."""
    sid = session.get("session_id")
    if sid:
        stale = [vid for vid, info in locked_videos.items() if info["session_id"] == sid]
        for vid in stale:
            del locked_videos[vid]
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes — main app
# ---------------------------------------------------------------------------


@app.route("/")
@login_required
def index():
    """Main annotation page."""
    return render_template(
        "index.html",
        default_labels=current_labels,
        annotator_name=session["annotator_name"],
    )


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.route("/api/next-video", methods=["POST"])
@login_required
def next_video():
    """Return the next video that needs annotation by the current user."""
    annotator_name = session["annotator_name"]
    sid = session["session_id"]

    video = get_next_video(annotator_name, sid)

    if video:
        entries = video.get("annotation_entries", [])
        target = video.get("target_annotations", 2)
        return jsonify(
            {
                "success": True,
                "video": video,
                "annotation_number": len(entries) + 1,
                "target_annotations": target,
            }
        )
    return jsonify({"success": False, "message": "No more videos to annotate!"})


@app.route("/api/annotate", methods=["POST"])
@login_required
def annotate():
    """Append this user's annotation entry to the video's annotation_entries list."""
    data_req = request.json
    video_id = data_req.get("video_id")
    annotations = data_req.get("annotations", [])
    annotator_name = session["annotator_name"]
    sid = session["session_id"]

    if not video_id:
        return jsonify({"success": False, "message": "No video ID provided"})

    with data_lock:
        data = load_data()

        for video in data:
            if video.get("video_id") != video_id:
                continue

            # Prevent the same person annotating the same video twice
            if any(e.get("annotated_by") == annotator_name
                   for e in video.get("annotation_entries", [])):
                unlock_video(video_id, sid)
                return jsonify(
                    {"success": False, "message": "You have already annotated this video."}
                )

            video.setdefault("annotation_entries", []).append(
                {
                    "annotations": annotations,
                    "annotated_at": datetime.now().isoformat(),
                    "annotated_by": annotator_name,
                }
            )
            break

        escalated = maybe_escalate_targets(data)
        save_data(data)
        unlock_video(video_id, sid)

    return jsonify({"success": True, "escalated": escalated})


@app.route("/api/skip", methods=["POST"])
@login_required
def skip_video():
    """Skip the current video without annotating (releases the lock)."""
    video_id = request.json.get("video_id")
    if video_id:
        unlock_video(video_id, session["session_id"])
    return jsonify({"success": True})


@app.route("/api/stats")
@login_required
def stats():
    """Return aggregate annotation progress statistics."""
    data = load_data()

    total = sum(
        1 for v in data
        if os.path.exists(os.path.join(VIDEO_DIR, f"{v.get('video_id')}.mp4"))
    )
    fully_done = sum(
        1 for v in data
        if len(v.get("annotation_entries", [])) >= v.get("target_annotations", 2)
        and os.path.exists(os.path.join(VIDEO_DIR, f"{v.get('video_id')}.mp4"))
    )
    partial = sum(
        1 for v in data
        if 0 < len(v.get("annotation_entries", [])) < v.get("target_annotations", 2)
        and os.path.exists(os.path.join(VIDEO_DIR, f"{v.get('video_id')}.mp4"))
    )
    double_annotated = sum(
        1 for v in data if len(v.get("annotation_entries", [])) >= 2
    )
    current_target = max(
        (v.get("target_annotations", 2) for v in data), default=2
    )

    return jsonify(
        {
            "total": total,
            "fully_done": fully_done,
            "partial": partial,
            "remaining": total - fully_done,
            "locked": len(locked_videos),
            "double_annotated": double_annotated,
            "current_target": current_target,
        }
    )


@app.route("/api/all-annotations")
@login_required
def all_annotations():
    """
    Return all videos annotated by the current user, with only their own
    annotation entry exposed (for the Edit tab).
    """
    annotator_name = session["annotator_name"]
    data = load_data()

    result = []
    for video in data:
        user_entries = [
            e for e in video.get("annotation_entries", [])
            if e.get("annotated_by") == annotator_name
        ]
        if not user_entries:
            continue
        # Strip annotation_entries; expose only this user's most recent entry
        v = {k: val for k, val in video.items() if k != "annotation_entries"}
        v["my_entry"] = user_entries[-1]
        result.append(v)

    # Sort by the user's annotation time, most recent first.
    # Fall back to empty string so entries without a timestamp sort to the end.
    result.sort(key=lambda v: v["my_entry"].get("annotated_at") or "", reverse=True)

    return jsonify(result)


@app.route("/api/update-annotation", methods=["POST"])
@login_required
def update_annotation():
    """Update the current user's annotation entry for a given video."""
    data_req = request.json
    video_id = data_req.get("video_id")
    annotations = data_req.get("annotations", [])
    annotator_name = session["annotator_name"]

    if not video_id:
        return jsonify({"success": False, "message": "No video ID provided"})

    with data_lock:
        data = load_data()

        for video in data:
            if video.get("video_id") != video_id:
                continue
            for entry in video.get("annotation_entries", []):
                if entry.get("annotated_by") == annotator_name:
                    entry["annotations"] = annotations
                    entry["last_edited_at"] = datetime.now().isoformat()
                    break
            break

        save_data(data)

    return jsonify({"success": True})


@app.route("/api/labels", methods=["GET", "POST"])
@login_required
def labels():
    """Get or update the annotation labels (persisted to LABELS_FILE)."""
    global current_labels
    if request.method == "GET":
        return jsonify(current_labels)
    current_labels = request.json.get("labels", current_labels)
    save_labels(current_labels)
    return jsonify({"success": True})


@app.route("/video/<path:filename>")
@login_required
def serve_video(filename):
    """Serve video files from the dataset directory."""
    return send_from_directory(VIDEO_DIR, filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)

    print("Running data migration check...")
    migrate_data()

    print("Video Annotation App starting...")
    print(f"  Data file  : {DATA_FILE}")
    print(f"  Video dir  : {VIDEO_DIR}")
    print("  Open http://localhost:5000 in your browser")

    app.run(debug=False, host="0.0.0.0", port=5000)
