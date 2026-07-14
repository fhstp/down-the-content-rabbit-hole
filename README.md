# Down the Content Rabbit Hole: Making Algorithmic Content Shift on Short-Form Video Platforms Measurable

Code accompanying the *Down the Content Rabbit Hole* paper. The repository covers the full pipeline:

1. **Scraping** YouTube Shorts via the YouTube Data API + `yt-dlp`
2. **Annotating** videos through a multi-annotator Flask web app
3. **Classifying** videos with zero-shot CLIP
4. **Evaluating** classifier outputs against human annotations
5. **Summarizing** dataset statistics for the paper

## AI Disclosure

The main classification algorithm, data retrieval as well as the core evaluation logic was written manually. The annotation app, "book keeping" in json and this README was done in large parts by Claude (Anthropic, Opus 4.6, Opus 4.8 and Fable 5) and verified by the authors. Claude Fable 5 was also used to do a final check of the content of this repository.

## Repository layout

```
scraping/         YouTube Shorts collection
annotation/       Flask annotation app + data merge utility
classification/   CLIP classification, evaluation, hyperparameter tuning
helpers/          Dataset and annotation statistics
```

All scripts resolve dataset paths relative to the repository root, so they can be invoked from any working directory. Intermediate outputs of the classification scripts (`clip_prompts.json`, `clip_embeddings/`, results and tuning files) are written to the current working directory unless overridden with the flags shown below.

## Required runtime files (not tracked in git)

```
<repo_root>/
  .env                                 YOUTUBE_API_KEY (and optionally SECRET_KEY, ANNOTATION_PASSWORD)
  classes.json                         class definitions consumed by classify_clip.py and tune_clip_hyperparams.py
  cookies.txt                          (optional) yt-dlp cookies for authenticated downloads
  dataset/
    yt_data.json                       master dataset (videos + metadata + annotations)
    annotation_labels.json             (auto-written) label set edited via the app's Settings tab
    videos/{video_id}.mp4              downloaded clips
    video_data_{class}_{i}.json        per-query scraper output (intermediate)
  classification/
    clip_prompts.json                  (optional) prompt banks, written by --prompt-mode llm
    clip_embeddings/                   CLIP frame-embedding cache
    best_params.json                   per-class CLIP hyperparameters from tuning
    classification_results_clip.json   CLIP classifier outputs
```

Templates for the two config files are tracked in git — copy and adapt them:

```bash
cp classes.example.json classes.json
cp .env.example .env      # then fill in the values
```

`classes.json` maps each class name to three plain strings; `include` / `exclude` are comma- or semicolon-separated phrases that are expanded into CLIP prompts (see `classes.example.json` for a complete two-class example):

```json
{
  "Animal": {
    "definition": "Videos featuring real animals such as pets, wildlife or livestock.",
    "include": "cats, dogs, wild animals, farm animals, veterinary care",
    "exclude": "animated animals, animal logos, humans in costumes"
  }
}
```

## Setup

Python 3.11+ recommended. Install requirements per stage:

```bash
# Scraping
pip install -r scraping/requirements.txt
# yt-dlp must also be on PATH (the package above provides it)

# Annotation
pip install -r annotation/requirements.txt

# Classification (CLIP)
pip install -r classification/requirements.txt
```

CLIP classification runs on CPU as well as GPU (pass `--device cpu`); the `ViT-B/32` checkpoint (~340 MB) is downloaded automatically on first use.

### Environment variables (`<repo_root>/.env`)

| Variable | Used by | Purpose |
|----------|---------|---------|
| `YOUTUBE_API_KEY` | scraping | YouTube Data API v3 key (required) |
| `SECRET_KEY` | annotation | Flask session secret (recommended; sessions reset on restart without it) |
| `ANNOTATION_PASSWORD` | annotation | Shared annotator password (defaults to `changeme`) |

## Quickstart without a YouTube API key

A synthetic demo dataset lets you try every stage except scraping (requires `ffmpeg` on PATH):

```bash
python helpers/make_demo_dataset.py          # writes dataset/ with 10 test clips + annotations
cp classes.example.json classes.json

python annotation/annotation_app.py          # annotate at http://localhost:5000 (password: changeme)
python classification/classify_clip.py --device cpu   # classify (CLIP downloads on first run)
python classification/evaluate.py --clip-results classification_results_clip.json
python helpers/summarize_dataset.py          # dataset + agreement statistics
```

The demo clips are colour test patterns, so CLIP classifies them all as negative — the point is to exercise the pipeline mechanics. Drop real `.mp4` files into `dataset/videos/` (named `{video_id}.mp4`) to see real scores.

## Pipeline

### 1. Scrape

```bash
python scraping/scrape_yt_shorts.py
```

Iterates over a hard-coded `classes_queries` dict (Animal, Car). For each query it retrieves video metadata via the YouTube Data API, skips live streams and videos longer than one minute, downloads the clips with `yt-dlp` (9:16 format only), and writes `dataset/videos/{video_id}.mp4` plus `dataset/video_data_{class}_{i}.json`. Edit the dict in the script to change classes or queries.

### 2. Merge per-query JSON files into the master dataset

```bash
python annotation/merge_data.py dataset/video_data_Animal_0.json
```

Appends new videos to the master file, deduplicating by `video_id`; existing entries are never modified, so re-running is safe. `--master` defaults to `<repo_root>/dataset/yt_data.json`.

### 3. Annotate

```bash
python annotation/annotation_app.py
```

Serves on `http://0.0.0.0:5000`. Annotators log in with the shared password and a personal name, then label one random video at a time (or skip). Features:

- **Multi-annotation**: each video is annotated by multiple people; the same person cannot annotate a video twice. Videos are locked while someone is working on them (1 h timeout).
- **Auto-escalation**: once 100 videos have 2 annotations, the target is raised to 3.
- **Review**: the *All annotations* view lets annotators revisit and edit their own entries.
- **Labels**: the label set can be edited at runtime via the Settings tab; edits are persisted to `dataset/annotation_labels.json` and survive restarts.

Annotations are stored per annotator in `annotation_entries` in `yt_data.json`.

### 4. Classify

```bash
python classification/classify_clip.py \
  --model ViT-B/32 \
  --prompt-mode auto \
  --classes-config classes.json \
  --embedding-cache classification/clip_embeddings \
  --output classification/classification_results_clip.json
```

Samples frames at 4 fps (max 32 per video), scores three spatial crops per frame against positive/negative prompt banks, and aggregates per-frame scores with a sliding window (see `--help` for `--per-frame-cutoff`, `--window-size`, `--min-positives-in-window`, `--threshold`, and the `mean` / `max` aggregation modes). Results are written after every video, so interrupted runs resume where they left off; use `--overwrite` to start fresh, and `--test` to process only a handful of videos.

Frame embeddings are cached under `--embedding-cache` (keyed by model, fps, and max-frames), so repeat runs and hyperparameter tuning skip the expensive encoding step.

Prompt banks (`--prompt-mode`):

- `auto` — expand class name + `classes.json` fields through fixed templates (no extra dependencies)
- `class-name` — fixed templates from the class name only
- `llm` — generate prompts via a local [Ollama](https://ollama.com) server (`--llm-host http://localhost:11434`, `--llm-model gemma3:27b`), save them to `--prompt-config`, then exit; re-run to classify with the generated prompts
- `config` — load prompts from `--prompt-config` (default when the file exists)

### 5. Tune CLIP hyperparameters (optional)

```bash
python classification/tune_clip_hyperparams.py \
  --classes-config classes.json \
  --embedding-cache classification/clip_embeddings \
  --best-params classification/best_params.json
```

Grid-searches the sliding-window parameters per class against the annotated subset, maximising F1. Run step 4 first so the embedding cache is warm (missing embeddings are computed on the fly). The grid is configurable via `--top-k-values`, `--per-frame-cutoff-values`, `--window-size-values`, `--min-positives-fractions`, and `--threshold-values`. Outputs: full grid (`tune_results.json`), per-class best (`best_params.json`), and global best (`best_params_global.json`).

Apply the tuned per-class settings by re-running classification:

```bash
python classification/classify_clip.py \
  --classes-config classes.json \
  --embedding-cache classification/clip_embeddings \
  --best-params-config classification/best_params.json \
  --output classification/classification_results_clip.json --overwrite
```

### 6. Evaluate

```bash
python classification/evaluate.py \
  --clip-results classification/classification_results_clip.json \
  --latex
```

Compares classifier outputs against human annotations. Ground truth per video is the majority vote across its `annotation_entries`: a label counts if strictly more than half of the annotators assigned it (with two annotators, both must agree). Prints confusion matrices and accuracy / precision / recall / F1 per (model, class), plus a summary table; `--latex` adds LaTeX (booktabs) tables, `--show-ids` lists false positives/negatives per video.

### 7. Dataset statistics

```bash
python helpers/summarize_dataset.py     # paper tables: overview, metadata, coverage,
                                        # label confusion, inter-annotator agreement (Cohen's kappa)
python helpers/analyze_annotations.py   # quick annotation coverage + label distribution
```

`summarize_dataset.py` prints every table both as plain text and as a LaTeX tabular block.

## Known caveats

- **Scraping** assumes `yt-dlp` is on PATH. Cookie-based authenticated downloads require `cookies.txt` at the repo root and a JS runtime such as `deno`; the call site is currently commented out in `scraping/scrape_yt_shorts.py`.
- **`--prompt-mode llm`** requires a running Ollama server; without one the script aborts with a connection error.

## Citation

If you use this code or build on the pipeline, please cite this repository and the work presented at the International Conference on Social Media & Society (SMS 2026):

<!-- TODO: fill in the repository URL and the slides link once published -->
```bibtex
@misc{kietreiber2026rabbithole,
  title        = {Down the Content Rabbit Hole: Making Algorithmic Content Shift on Short-Form Video Platforms Measurable},
  author       = {Kietreiber, Tobias and Krejca, Felix and Neubacher, Anna and Neumair, Sebastian and Klausner, Lukas-Daniel},
  year         = {2026},
  howpublished = {\url{https://github.com/fhstp/down-the-content-rabbit-hole}},
  note         = {Code repository. Presented at the International Conference on Social Media \& Society (SMS 2026)},
}
```
