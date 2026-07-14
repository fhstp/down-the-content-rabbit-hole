"""classify_clip.py — CLIP-based video classifier
(via the OpenAI clip / PyTorch package).

Inference pipeline (per frame):
  1. Preprocess: resize → 3 spatial crops (top / center / bottom) → (3, 3, 224, 224)
  2. Encode:     model.encode_image(crops) → (3, D) L2-normalised embeddings
  3. Score:      for each prompt, max cosine-sim across the 3 crops;
                 top-K mean over positives and negatives;
                 softmax(100 × [pos_mean, neg_mean]) → (pos_prob, neg_prob)
  4. Aggregate:  sliding-window (default), mean, or max of per-frame pos_prob → video score
  5. Threshold:
       sliding-window/relative: fraction of active frames > --threshold (default 0.5)
       sliding-window/absolute: active frame count >= --count-threshold
       mean / max:              aggregated score > --threshold (default 0.5)

     --threshold and --count-threshold are mutually exclusive; setting both raises an error.
"""

import argparse
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = str(REPO_ROOT / "dataset" / "yt_data.json")
DEFAULT_DATASET_ROOT = str(REPO_ROOT / "dataset")

import clip
import numpy as np
import requests
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from tqdm.rich import tqdm


# ── Default generic (background / negative) prompt bank ──────────────────────
# Appended to every class's negative side unless overridden by a '_generic'
# key in the prompt config.
DEFAULT_GENERIC_PROMPTS: List[str] = [
    # Human-Centric
    "a person",
    "a photo of a person",
    "a child",
    "a crowd of people",
    "a group of children",
    "an elderly person",
    "a group of old people",
    "a selfie",
    "a face",
    "a vlogger",
    "a person talking to the camera",
    "a makeup tutorial",
    "a fitness workout",
    "people dancing",
    "a news anchor",
    "a tattooed person",
    "a tattooed body part",
    # Indoor / Domestic
    "an empty room",
    "a living room",
    "a bedroom",
    "a kitchen",
    "furniture",
    "a messy room",
    "an office",
    "a white wall",
    "ceiling",
    "floor",
    # Digital / Screen Content
    "a screenshot",
    "text on screen",
    "a meme",
    "a computer screen",
    "a phone screen",
    # Urban / Outdoor
    "a street",
    "a building",
    "city life",
    "architecture",
    "scenery",
    "landscape",
    "sky",
    "a wooden log",
    "wood",
    "a wooden stick",
    # Indoor/Domestic (detailed)
    "a photo of a cozy living room with a sofa",
    "a clean modern kitchen with marble countertops",
    "a messy bedroom with an unmade bed",
    "a dimly lit hallway in an apartment",
    "a bathroom with a tiled shower and mirror",
    "a home office setup with a computer and desk",
    "an empty dining room with a wooden table",
    "a laundry room with a washing machine",
    "a sunlit nursery with a white crib",
    "a basement storage area with cardboard boxes",
    # Urban / Architecture
    "a busy city street with cars and traffic",
    "a crowded sidewalk with pedestrians walking",
    "a neon-lit city street at night",
    "a view of skyscrapers against a blue sky",
    "a quiet alleyway with brick walls",
    "a subway station platform with tracks",
    "a construction site with cranes and scaffolding",
    "an outdoor cafe with people sitting at tables",
    "a brick wall with graffiti",
    # Nature / Landscapes
    "a lush green forest with tall trees",
    "a wide open field of tall grass",
    "a snow-capped mountain range",
    "a calm blue ocean with small waves",
    "a sandy beach with palm trees",
    "a dry desert landscape with sand dunes",
    "a rocky canyon under a bright sun",
    "a peaceful lake reflecting the sky",
    "a flowing river or stream in the woods",
    "a garden filled with colorful flowers",
    # Commercial / Public Spaces
    "the interior of a grocery store aisle",
    "a modern office open-plan workspace",
    "a high-end clothing boutique interior",
    "a dark industrial warehouse with high ceilings",
    "a busy restaurant kitchen with chefs",
    "a hospital waiting room with chairs",
    "a library with rows of bookshelves",
    "a gym with weightlifting equipment",
    "a gas station at night with bright lights",
    "a movie theater with red velvet seats",
    # Abstract / Miscellaneous
    "a close-up of a blank white wall",
    "an abstract pattern of colors and shapes",
    "a blurred background with bokeh lights",
    "a rainy window with water droplets",
    "a top-down view of a wooden floor",
    "a macro shot of a fabric texture",
    "a foggy morning in a rural area",
    "a starry night sky with the milky way",
    "a stack of old books on a table",
    "a flat lay of office stationery",
]


# ── PyTorch CLIP helpers ──────────────────────────────────────────────────────

def _make_3crops(frame_np: np.ndarray, target_size: int = 224) -> List[Image.Image]:
    """Resize so shorter side == target_size, then return top / center / bottom crops."""
    img = Image.fromarray(frame_np)
    w, h = img.size
    scale = target_size / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left   = (new_w - target_size) // 2
    c_top  = (new_h - target_size) // 2

    top_crop    = img.crop((left, 0,                      left + target_size, target_size))
    center_crop = img.crop((left, c_top,                  left + target_size, c_top + target_size))
    bottom_crop = img.crop((left, new_h - target_size,    left + target_size, new_h))
    return [top_crop, center_crop, bottom_crop]


def preprocess_frame(
    frame_np: np.ndarray,
    preprocess_fn,
    device: torch.device,
) -> torch.Tensor:
    """(H, W, 3) uint8 numpy → (3, 3, 224, 224) float32 tensor on device."""
    crops = _make_3crops(frame_np)
    tensors = [preprocess_fn(c) for c in crops]
    return torch.stack(tensors).to(device)  # (3, C, 224, 224)


def score_frame(
    image_embed: torch.Tensor,    # (3, D)   — one embedding per crop
    positive_embed: torch.Tensor, # (N+, D)
    negative_embed: torch.Tensor, # (N-, D)
    top_k: int = 10,
) -> Tuple[float, float]:
    """Score one pre-encoded frame (3 crops) against positive / negative banks.

    For each prompt the max cosine-similarity across the 3 crops is taken,
    then the top-K similarities are averaged, and finally softmax(100 × ...)
    is applied.

    Returns:
        (pos_prob, neg_prob) as Python floats.
    """
    # L2-normalise (CLIP encode_* already does this, but guard anyway)
    img_n = F.normalize(image_embed,    dim=-1)  # (3, D)
    pos_n = F.normalize(positive_embed, dim=-1)  # (N+, D)
    neg_n = F.normalize(negative_embed, dim=-1)  # (N-, D)

    # cosine sims via dot product: (3, N+) and (3, N-)
    sims_pos = img_n @ pos_n.T  # (3, N+)
    sims_neg = img_n @ neg_n.T  # (3, N-)

    # max across the 3 spatial crops → (N+,) and (N-,)
    max_pos = sims_pos.max(dim=0).values
    max_neg = sims_neg.max(dim=0).values

    k_pos = max_pos.shape[0] if top_k == -1 else min(top_k, max_pos.shape[0])
    k_neg = max_neg.shape[0] if top_k == -1 else min(top_k, max_neg.shape[0])

    mean_pos = max_pos.topk(k_pos).values.mean()
    mean_neg = max_neg.topk(k_neg).values.mean()

    probs = torch.softmax(100.0 * torch.stack([mean_pos, mean_neg]), dim=0)
    return float(probs[0]), float(probs[1])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_model_key(model_name: str) -> str:
    """'ViT-B/32' → 'clip_vitb32' for use as a results JSON key."""
    clean = re.sub(r"[^a-zA-Z0-9]", "", model_name).lower()
    return f"clip_{clean}"


# ── Prompt system ─────────────────────────────────────────────────────────────

def build_auto_prompts(
    class_name: str,
    class_config: dict,
    generic_prompts: List[str],
) -> Dict[str, List[str]]:
    """Generate prompt banks from classes.json fields using template expansion."""
    cls_lower  = class_name.lower()
    definition = class_config.get("definition", "")
    include    = class_config.get("include",    "")
    exclude    = class_config.get("exclude",    "")

    # Positive: class-name templates + definition sentence + include excerpts.
    positive: List[str] = [
        f"a video of {cls_lower}",
        f"a short video featuring {cls_lower}",
        f"a video that shows {cls_lower}",
        f"a clip with {cls_lower}",
        f"{cls_lower} in a video",
        f"a photo of {cls_lower}",
    ]
    defn_sentence = definition.split(".")[0].strip()
    if defn_sentence:
        positive.append(f"a video of {defn_sentence.lower()}")
    for part in re.split(r"[,;]", include):
        part = part.strip().rstrip(".")
        if part and len(part) < 80:
            positive.append(f"a video featuring {part.lower()}")

    # Negative: generic bank + exclude excerpts.
    negative: List[str] = list(generic_prompts)
    for part in re.split(r"[,;]", exclude):
        part = part.strip().rstrip(".")
        if part and len(part) < 80:
            negative.append(f"a video of {part.lower()}")

    return {"positive": positive, "negative": negative}


def build_classname_prompts(
    class_name: str,
    generic_prompts: List[str],
) -> Dict[str, List[str]]:
    """Minimal prompt banks using only the class name in fixed templates."""
    cls_lower = class_name.lower()
    return {
        "positive": [
            f"a video of {cls_lower}",
            f"a short video featuring {cls_lower}",
            f"a video that shows {cls_lower}",
            f"a clip with {cls_lower}",
            f"{cls_lower} in a video",
            f"a photo of {cls_lower}",
        ],
        "negative": list(generic_prompts),
    }


def generate_llm_prompts(
    class_name: str,
    class_config: dict,
    n: int = 20,
    llm_model: str = "gemma3:27b",
    host: str = "http://localhost:11434",
) -> Dict[str, List[str]]:
    """Call Ollama to generate positive and negative CLIP prompt banks."""
    defn    = class_config.get("definition", "")
    include = class_config.get("include",    "")
    exclude = class_config.get("exclude",    "")

    system_msg = (
        "You are an expert at zero-shot image/video classification using CLIP. "
        "Your task is to generate short, concrete, visually descriptive text prompts "
        "that maximise cosine similarity with relevant image embeddings."
    )
    user_msg = (
        f"Generate exactly {n} POSITIVE and {n} NEGATIVE CLIP prompts for class: '{class_name}'.\n\n"
        f"Definition: {defn}\n"
        f"Include (positive examples): {include}\n"
        f"Exclude (not this class): {exclude}\n\n"
        "POSITIVE prompts: describe what a video frame looks like when this class IS present.\n"
        "NEGATIVE prompts: describe frames where this class is NOT present.\n"
        "Keep each prompt under 15 words. Be specific and visually concrete.\n\n"
        "Respond with ONLY valid JSON, no explanation:\n"
        '{"positive": ["...", ...], "negative": ["...", ...]}'
    )

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "format": "json",
    }
    url = f"{host.rstrip('/')}/api/chat"
    print(f"  Ollama ({llm_model}): generating prompts for '{class_name}'...")
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed for '{class_name}': {e}") from e

    raw = resp.json()["message"]["content"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            raise ValueError(f"Could not parse LLM JSON for '{class_name}': {raw[:500]}")

    return {
        "positive": list(data.get("positive", [])),
        "negative": list(data.get("negative", [])),
    }


def build_prompt_banks(
    class_names: List[str],
    classes_config: Dict[str, dict],
    prompt_mode: str,
    prompt_config_path: str,
    llm_n: int = 20,
    llm_model: str = "gemma3:27b",
    llm_host: str = "http://localhost:11434",
) -> Dict[str, Dict[str, List[str]]]:
    """Build, load, or generate prompt banks for all classes.

    Returns: {class_name: {"positive": [...], "negative": [...]}}

    When prompt_mode == 'llm': generates via Ollama, saves to prompt_config_path,
    and returns the banks (caller should exit afterwards if desired).
    """
    generic = DEFAULT_GENERIC_PROMPTS

    # ── config ────────────────────────────────────────────────────────────────
    if prompt_mode == "config":
        print(f"Loading prompts from {prompt_config_path}")
        with open(prompt_config_path) as f:
            cfg = json.load(f)
        if "_generic" in cfg:
            generic = cfg["_generic"]
        banks: Dict[str, Dict[str, List[str]]] = {}
        for cls in class_names:
            if cls not in cfg:
                raise KeyError(
                    f"Class '{cls}' not found in {prompt_config_path}. "
                    "Re-run with --prompt-mode llm or --prompt-mode auto to generate."
                )
            banks[cls] = cfg[cls]
        return banks

    # ── llm ───────────────────────────────────────────────────────────────────
    if prompt_mode == "llm":
        print(f"Generating prompts via Ollama ({llm_model}) for {len(class_names)} classes...")
        banks = {}
        for cls in class_names:
            banks[cls] = generate_llm_prompts(
                cls, classes_config[cls], n=llm_n, llm_model=llm_model, host=llm_host,
            )
        cfg = {"_generic": generic, **banks}
        with open(prompt_config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Saved prompt config to {prompt_config_path}")
        return banks

    # ── auto / class-name ─────────────────────────────────────────────────────
    banks = {}
    for cls in class_names:
        if prompt_mode == "auto":
            banks[cls] = build_auto_prompts(cls, classes_config[cls], generic)
        else:  # class-name
            banks[cls] = build_classname_prompts(cls, generic)
    return banks


# ── Embedding cache ──────────────────────────────────────────────────────────

def _embedding_cache_path(
    cache_dir: str,
    model_key: str,
    fps: float,
    max_frames: int,
    video_id: str,
) -> str:
    """Return the .npz path for a given video's cached frame embeddings.

    Directory structure: {cache_dir}/{model_key}/fps{fps}_maxf{max_frames}/{video_id}.npz
    Embedding model and sampling parameters are encoded in the path so
    different settings never share a cache entry.
    """
    fps_tag = f"{fps:g}"   # e.g. 4.0 → "4", 2.5 → "2.5"
    return os.path.join(
        cache_dir, model_key, f"fps{fps_tag}_maxf{max_frames}", f"{video_id}.npz"
    )


def load_cached_embeddings(
    cache_path: str,
) -> Optional[Tuple[List[np.ndarray], List[float]]]:
    """Load cached frame embeddings from *cache_path*.

    Returns:
        (embeds_list, timestamps) where embeds_list is a list of (3, D)
        float32 numpy arrays (one per frame), or None if not cached /
        cache is corrupt.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path)
        embeds_np  = data["embeds"]     # (num_frames, 3, D) float32
        timestamps = data["timestamps"].tolist()
        embeds_list = [embeds_np[i] for i in range(len(embeds_np))]
        return embeds_list, timestamps
    except Exception:
        return None  # corrupt cache — will be re-computed and overwritten


def save_cached_embeddings(
    cache_path: str,
    embeds_list: List[np.ndarray],   # list of (3, D) float32 arrays
    timestamps: List[float],
) -> None:
    """Persist frame embeddings to *cache_path* as a compressed .npz."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeds    = np.stack(embeds_list).astype(np.float32),  # (N, 3, D)
        timestamps= np.array(timestamps, dtype=np.float64),
    )


# ── Video loading ─────────────────────────────────────────────────────────────

def load_video_frames(
    video_path: str,
    fps: float = 4.0,
    max_frames: int = 32,
) -> Tuple[List[np.ndarray], List[float]]:
    """Uniformly sample frames from a video using decord.

    Returns:
        frames:       list of (H, W, 3) uint8 RGB numpy arrays
        timestamps_s: corresponding timestamps in seconds
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    video_fps    = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration     = total_frames / video_fps

    num_samples = max(1, min(int(duration * fps), max_frames))
    indices     = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames     = [vr[int(idx)].asnumpy() for idx in indices]
    timestamps = [round(int(idx) / video_fps, 4) for idx in indices]
    return frames, timestamps


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Validate and resolve threshold / count-threshold ─────────────────────
    if args.aggregation == "sliding-window":
        if args.threshold is not None and args.count_threshold is not None:
            raise SystemExit(
                "error: --threshold and --count-threshold are mutually exclusive; "
                "set at most one."
            )
        _use_relative = args.count_threshold is None
        _threshold    = args.threshold if _use_relative else None
        if _use_relative and _threshold is None:
            _threshold = 0.5   # built-in default for relative mode
    else:
        if args.count_threshold is not None:
            raise SystemExit(
                "error: --count-threshold is only valid with --aggregation sliding-window."
            )
        _use_relative = True   # not used in mean/max path, but keeps linter happy
        _threshold    = args.threshold if args.threshold is not None else 0.5

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) != args.device:
        print(f"Warning: CUDA not available; falling back to {device}")

    # ── Class config ─────────────────────────────────────────────────────────
    with open(args.classes_config) as f:
        classes_config = json.load(f)

    class_names = (
        [c.strip() for c in args.classes.split(",")]
        if args.classes
        else list(classes_config.keys())
    )

    # ── Best params per class (from tune_clip_hyperparams.py) ─────────────────
    best_params_per_class: Dict[str, dict] = {}
    if args.best_params_config:
        with open(args.best_params_config) as f:
            _bp_data = json.load(f)
        print(f"Loading per-class best params from {args.best_params_config}")
        for cls in class_names:
            if cls in _bp_data:
                best_params_per_class[cls] = _bp_data[cls]
                bp = _bp_data[cls]
                print(
                    f"  {cls}: top_k={bp['top_k']}  "
                    f"cutoff={bp['per_frame_cutoff']}  "
                    f"ws={bp['window_size']}  "
                    f"minW={bp['min_positives_in_window']}  "
                    f"threshold={bp['threshold']}  "
                    f"(tuned F1={bp.get('f1', '?'):.4f})"
                )
            else:
                print(f"  Warning: '{cls}' not found in best params; using CLI defaults")

    # ── Dataset ──────────────────────────────────────────────────────────────
    with open(args.dataset) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} videos, {len(class_names)} classes")

    # ── Prompt mode ──────────────────────────────────────────────────────────
    prompt_mode = args.prompt_mode
    if prompt_mode is None:
        prompt_mode = "config" if os.path.exists(args.prompt_config) else "auto"
        print(f"--prompt-mode not set; using '{prompt_mode}'")

    prompt_banks = build_prompt_banks(
        class_names,
        classes_config,
        prompt_mode,
        args.prompt_config,
        llm_n=args.llm_n,
        llm_model=args.llm_model,
        llm_host=args.llm_host,
    )

    # LLM mode: prompts have been saved; exit before loading the CLIP model.
    if prompt_mode == "llm":
        print("Prompt generation complete. Re-run to classify (prompts will be loaded from config).")
        return

    # ── Load CLIP model ───────────────────────────────────────────────────────
    # CLIP uses a causal attention mask; newer cuDNN SDP backends can choke on it.
    # Force the math (unfused) backend to avoid RuntimeError from cuDNN Frontend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    print(f"Loading CLIP {args.model} on {device}...")
    model, preprocess_fn = clip.load(args.model, device=device)
    # CLIP loads in float16 on CUDA but its causal attn_mask is float32,
    # causing a cuDNN dtype mismatch with newer PyTorch/cuDNN.  Convert
    # the whole model to float32 to guarantee consistency.
    model = model.float()
    model.eval()
    print("Model loaded.")

    # ── Encode text embeddings ────────────────────────────────────────────────
    print("Encoding text prompt banks...")
    text_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for cls in class_names:
            pos_prompts = prompt_banks[cls]["positive"]
            neg_prompts = prompt_banks[cls]["negative"]
            print(f"  '{cls}': {len(pos_prompts)} positive, {len(neg_prompts)} negative prompts")
            pos_tokens = clip.tokenize(pos_prompts).to(device)
            neg_tokens = clip.tokenize(neg_prompts).to(device)
            text_embeddings[cls] = {
                "positive": model.encode_text(pos_tokens).float(),  # (N+, D)
                "negative": model.encode_text(neg_tokens).float(),  # (N-, D)
            }

    model_key = _sanitize_model_key(args.model)

    # ── Load / init results ───────────────────────────────────────────────────
    if not args.overwrite and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming from existing results in {args.output}")
    else:
        results = {}

    results.setdefault(model_key, {})
    for cls in class_names:
        results[model_key].setdefault(cls, {})
    results[model_key].setdefault("_timing",    {})
    results[model_key].setdefault("_frame_map", {})

    # ── Classification loop ───────────────────────────────────────────────────
    newly_processed = 0
    cache_hits = 0
    cache_misses = 0
    use_cache = bool(args.embedding_cache)

    for data in tqdm(dataset, desc=model_key):
        video_id   = data["video_id"]
        video_path = (
            os.path.join(args.dataset_root, data["local_filepath"])
            if data.get("local_filepath")
            else os.path.join(args.dataset_root, "videos", f"{video_id}.mp4")
        )

        # Skip if every class is already done for this video.
        if all(video_id in results[model_key].get(cn, {}) for cn in class_names):
            continue

        # ── Try embedding cache ───────────────────────────────────────────────
        cache_path = (
            _embedding_cache_path(
                args.embedding_cache, model_key, args.fps, args.max_frames, video_id
            )
            if use_cache else None
        )
        cached = load_cached_embeddings(cache_path) if cache_path else None

        if cached is not None:
            embeds_np_list, timestamps = cached
            frame_embeds = [
                torch.from_numpy(e).to(device) for e in embeds_np_list
            ]
            encode_s = 0.0
            cache_hits += 1

            # Restore _frame_map entry from cache if not already present.
            results[model_key]["_frame_map"].setdefault(
                video_id, {"num_frames": len(timestamps), "frame_timestamps_s": timestamps}
            )
        else:
            # ── Load frames from disk ─────────────────────────────────────────
            if not os.path.exists(video_path):
                print(f"Warning: video not found and no cache: {video_path}")
                for cn in class_names:
                    results[model_key][cn].setdefault(video_id, {"error": "video not found"})
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                continue

            try:
                frames, timestamps = load_video_frames(video_path, args.fps, args.max_frames)
            except Exception as e:
                print(f"Error loading frames for {video_id}: {e}")
                for cn in class_names:
                    results[model_key][cn].setdefault(video_id, {"error": str(e)})
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                continue

            results[model_key]["_frame_map"].setdefault(
                video_id, {"num_frames": len(frames), "frame_timestamps_s": timestamps}
            )

            # ── Encode frames ──────────────────────────────────────────────────
            try:
                t_encode = time.monotonic()
                frame_embeds: List[torch.Tensor] = []
                embeds_np_list: List[np.ndarray] = []
                with torch.no_grad():
                    for frame in frames:
                        preprocessed = preprocess_frame(frame, preprocess_fn, device)
                        embed = model.encode_image(preprocessed).float()   # (3, D)
                        frame_embeds.append(embed)
                        embeds_np_list.append(embed.cpu().numpy())
                encode_s = time.monotonic() - t_encode
                cache_misses += 1

                # Persist to cache so future runs skip encoding.
                if cache_path:
                    save_cached_embeddings(cache_path, embeds_np_list, timestamps)
            except Exception as e:
                print(f"Error encoding frames for {video_id}: {e}")
                for cn in class_names:
                    results[model_key][cn].setdefault(video_id, {"error": str(e)})
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                continue

        # ── Classify per class ────────────────────────────────────────────────
        per_class_s: Dict[str, float] = {}

        for cls in class_names:
            if video_id in results[model_key][cls]:
                continue  # already done — skip

            # ── Resolve per-class params (best-params override if loaded) ─────
            _bp_cls = best_params_per_class.get(cls, {})
            if _bp_cls:
                _cls_top_k           = _bp_cls["top_k"]
                _cls_cutoff          = _bp_cls["per_frame_cutoff"]
                _cls_ws              = _bp_cls["window_size"]
                _cls_minw            = _bp_cls["min_positives_in_window"]
                _cls_use_relative    = True
                _cls_threshold       = _bp_cls["threshold"]
                _cls_count_threshold = None
            else:
                _cls_top_k           = args.top_k
                _cls_cutoff          = args.per_frame_cutoff
                _cls_ws              = args.window_size
                _cls_minw            = args.min_positives_in_window
                _cls_use_relative    = _use_relative
                _cls_threshold       = _threshold
                _cls_count_threshold = args.count_threshold

            pos_embed = text_embeddings[cls]["positive"]
            neg_embed = text_embeddings[cls]["negative"]

            try:
                t_cls = time.monotonic()
                frame_pos_scores = [
                    score_frame(embed, pos_embed, neg_embed, top_k=_cls_top_k)[0]
                    for embed in frame_embeds
                ]
                scores_arr = np.array(frame_pos_scores)
                n_frames = len(frame_pos_scores)

                if args.aggregation == "sliding-window":
                    window = deque([False] * _cls_ws, _cls_ws)
                    active_count = 0
                    for score in frame_pos_scores:
                        is_pos = score > _cls_cutoff
                        window.append(is_pos)
                        if is_pos and sum(window) >= _cls_minw:
                            active_count += 1
                    if _cls_use_relative:
                        agg            = active_count / n_frames if n_frames else 0.0
                        classification = agg > _cls_threshold
                    else:
                        agg            = float(active_count)
                        classification = active_count >= _cls_count_threshold
                else:
                    agg = float(
                        scores_arr.mean() if args.aggregation == "mean" else scores_arr.max()
                    )
                    classification = agg > _cls_threshold

                per_class_s[cls] = time.monotonic() - t_cls

                if args.aggregation == "sliding-window":
                    if _cls_use_relative:
                        threshold_str = f"threshold: {_cls_threshold} (relative)"
                    else:
                        threshold_str = f"count_threshold: {_cls_count_threshold} (absolute)"
                    explanation = (
                        f"score: {agg:.4f} | {threshold_str} | "
                        f"aggregation: sliding-window | "
                        f"per_frame_cutoff: {_cls_cutoff} | "
                        f"window_size: {_cls_ws} | "
                        f"min_positives_in_window: {_cls_minw} | "
                        f"active_frames: {active_count}/{n_frames} | "
                        f"frames: {n_frames} | "
                        f"min: {scores_arr.min():.4f} max: {scores_arr.max():.4f}"
                    )
                else:
                    explanation = (
                        f"score: {agg:.4f} | threshold: {_cls_threshold} | "
                        f"aggregation: {args.aggregation} | "
                        f"frames: {n_frames} | "
                        f"min: {scores_arr.min():.4f} max: {scores_arr.max():.4f}"
                    )

                results[model_key][cls][video_id] = {
                    "classification": bool(classification),
                    "explanation":    explanation,
                }
            except Exception as e:
                print(f"Error classifying {video_id}/{cls}: {e}")
                results[model_key][cls][video_id] = {"error": str(e)}

        # ── Timing ────────────────────────────────────────────────────────────
        results[model_key]["_timing"][video_id] = {
            "encode_s":    round(encode_s, 3),
            "per_class_s": {k: round(v, 3) for k, v in per_class_s.items()},
            "total_s":     round(encode_s + sum(per_class_s.values()), 3),
        }

        # Persist after every video so the run is fully resumable.
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        newly_processed += 1
        if args.test and newly_processed >= args.test_size:
            print(f"[--test] Processed {newly_processed} videos. Stopping.")
            break

    if use_cache:
        print(
            f"\nEmbedding cache: {cache_hits} hits, {cache_misses} misses "
            f"(cache dir: {args.embedding_cache})"
        )
    print(f"Done. Results saved to {args.output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP-based video classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # CLIP model
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="OpenAI CLIP model variant (e.g. ViT-B/32, ViT-L/14)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string (cuda / cpu)",
    )

    # Prompt system
    parser.add_argument(
        "--prompt-mode",
        default=None,
        choices=["class-name", "auto", "llm", "config"],
        help=(
            "How to build text prompt banks. "
            "'class-name': fixed templates from the class name only. "
            "'auto': expand classes.json definition/include/exclude fields. "
            "'llm': generate via Ollama, save to --prompt-config, then exit. "
            "'config': load from --prompt-config JSON. "
            "Default: 'config' if --prompt-config file exists, else 'auto'."
        ),
    )
    parser.add_argument(
        "--prompt-config",
        default="clip_prompts.json",
        metavar="PATH",
        help="Prompt config JSON (written by --prompt-mode llm, read by --prompt-mode config)",
    )
    parser.add_argument(
        "--llm-n",
        type=int,
        default=20,
        metavar="N",
        help="Prompts per polarity per class for --prompt-mode llm",
    )
    parser.add_argument(
        "--llm-model",
        default="gemma3:27b",
        help="Ollama model name for --prompt-mode llm",
    )
    parser.add_argument(
        "--llm-host",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama host URL for --prompt-mode llm",
    )

    # Scoring
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="K",
        help="Top-K prompt similarities to average per frame (-1 = all)",
    )
    parser.add_argument(
        "--aggregation",
        choices=["sliding-window", "mean", "max"],
        default="sliding-window",
        help="How to aggregate per-frame positive scores into a video score",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Video-level decision threshold. "
            "sliding-window: fraction of active frames that must exceed this value (default 0.5). "
            "mean/max: aggregated score threshold (default 0.5). "
            "Mutually exclusive with --count-threshold."
        ),
    )
    parser.add_argument(
        "--count-threshold",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Absolute number of active frames required for a positive classification "
            "(sliding-window only). Mutually exclusive with --threshold."
        ),
    )
    parser.add_argument(
        "--per-frame-cutoff",
        type=float,
        default=0.8,
        metavar="CUTOFF",
        help="Per-frame binary threshold applied to pos_prob (sliding-window only)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        metavar="W",
        help="Rolling window length for the sliding-window aggregation",
    )
    parser.add_argument(
        "--min-positives-in-window",
        type=int,
        default=8,
        metavar="M",
        help=(
            "Minimum number of positive frames in the last --window-size frames "
            "for the current frame to count as 'active' (sliding-window only)"
        ),
    )

    # Frame sampling
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Target frame sampling rate",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        metavar="N",
        help="Maximum frames sampled per video",
    )

    # Dataset / output
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to yt_data.json (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"Root directory for video files (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        default="classification_results_clip.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--classes-config",
        default="classes.json",
        help="Path to classes.json",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names to run (default: all classes in config)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run all videos, ignoring any existing results",
    )
    parser.add_argument(
        "--embedding-cache",
        default="clip_embeddings",
        metavar="DIR",
        help=(
            "Directory for cached frame embeddings (.npz, one file per video). "
            "Cache is keyed by model + fps + max-frames so changing those params "
            "automatically uses a separate cache.  Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--best-params-config",
        default=None,
        metavar="PATH",
        help=(
            "Path to best_params.json produced by tune_clip_hyperparams.py. "
            "When set, per-class parameters (top_k, per_frame_cutoff, window_size, "
            "min_positives_in_window, threshold) are loaded from this file and override "
            "the corresponding CLI flags for each class. "
            "Only applies to --aggregation sliding-window. "
            "Classes not present in the file fall back to CLI defaults."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Limit to --test-size newly-processed videos",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5,
        metavar="N",
        help="Number of videos to process in test mode",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
