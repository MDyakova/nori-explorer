# NoRI Explorer

A browser-based tool ([Dash](https://dash.plotly.com/)) for exploring the outputs of NoRI
aging-prediction models: UMAP embeddings, regression metrics, attention scores, per-tile
heatmaps, whole-image prediction overlays, raw multi-channel images, and quantitative
morphology features with statistics.

Ported from `interactive_tool_outputs_regression.ipynb`.

## Install

```bash
pip install -r app/requirements.txt
```

## Run

```bash
python app/app.py [--base-dir /path/to/data/root]
```

This opens a browser at `http://127.0.0.1:8050`. The **data folder** can also be typed
into the "Data folder" field in the UI at any time instead of passing `--base-dir` — it
defaults to the current working directory.

Other flags: `--host` (default `127.0.0.1`), `--port` (default `8050`), and
`--no-browser` (skip auto-opening a browser tab on start).

### Running on a remote machine

- **Just for yourself** (recommended — no extra flags needed): run the command above on
  the remote machine as normal, then tunnel the port over SSH from your local machine and
  open it locally:
  ```bash
  ssh -L 8050:localhost:8050 user@remote-host
  # then open http://127.0.0.1:8050 in your local browser
  ```
- **For others on the same trusted network/VPN** (no SSH tunnel): bind to all interfaces
  with `--host 0.0.0.0`, and skip the auto-opened browser with `--no-browser` on a
  headless machine:
  ```bash
  python app/app.py --base-dir /path/to/data/root --host 0.0.0.0 --port 8050 --no-browser
  ```
  Then open `http://<remote-machine-ip>:8050` from any machine on that network, and make
  sure the port is allowed through the remote machine's firewall. This runs Dash's
  built-in development server with no authentication or HTTPS, so only do this on a
  trusted private network/VPN — never expose it directly to the open internet.

## Expected data layout

The data root passed via `--base-dir` (or entered in the UI) must contain `outputs/` and,
for the whole-image overlay and image viewer, `data/`:

```
<base_dir>/
├── outputs/
│   └── <group>/
│       └── <task>/
│           ├── umap/
│           │   └── <model_name>.csv        # columns: age, filename, sample_name,
│           │                                #          umap1, umap2, prediction, ...
│           ├── max_values.csv               # columns: protein, lipid
│           ├── channels.txt                 # optional, for the Image viewer's channel
│           │                                #   names/order (defaults to protein/lipid)
│           ├── heatmaps/
│           │   └── <model_name>/<age>/<filename>   # per-tile heatmap images
│           ├── models/                       # optional training-curve images, named
│           │   ├── epoch_loss_<model_name>.*        # e.g. loss_<model>.png
│           │   ├── mae_<model_name>.*
│           │   └── r2_<model_name>.*
│           └── features/                     # optional, for the "Features" page
│               ├── tubule_features.csv              # tubule-level features, every sample
│               ├── nuclei_features.csv               # per-nucleus features, every sample
│               └── shape_features.csv                # per-tubule shape features, every sample
└── data/
    └── <group>/
        └── <class_name>/
            └── <sample>_<class_name>_...tif  # whole-slide, multi-channel source images
```

This is separate from the app's own bundled assets under `app/assets/` (the header image
and the Feature dictionary / AI hypotheses documents below) — those ship with the app and
don't need to be in your data folder.

## Pages

- **Main page** — set the data folder, pick a group / task, and jump to any of the pages
  below. Also has an "About this model" explainer (what the multimodal model is, how the
  four modalities/inputs are fused, what its outputs mean) with a collapsible gallery of
  every reference diagram from the model-explanation deck.
- **UMAP explorer** ("Show UMAP") — pick a model, view its UMAP scatter plot, regression
  metrics (MAE, R²) and attention-score boxplots by age; click a point to inspect its tile
  heatmap.
- **All model plots** ("Show all model plots") — side-by-side UMAP, density, and
  prediction plots (plus training curves) for every model in a task, with a combined
  prediction-by-sample boxplot, an animal-level distribution-separation plot, and a
  median-predicted-age-by-animal plot.
- **Whole-image prediction overlay** ("Show prediction map") — pick a model and a
  whole-slide `.tif`, and view it with per-tile predictions overlaid as a heatmap. Supports
  click-to-zoom, and an "Inspect tile" mode that looks up and shows that tile's per-model
  heatmap(s).
- **Image viewer** — view a raw whole-slide `.tif` with any combination of its channels
  composited together (per-channel or global intensity normalization), click-to-zoom, and
  draw rectangle/polygon regions to plot each region's protein/lipid intensity
  distribution.
- **Features** — pick one of the three feature-file kinds (tubule, nuclei, shape) and an
  aggregation level (raw tubules, per-animal median/mean, or per-animal-and-tubule-type
  median), plus an optional filter (restrict to one or more values of any text/categorical
  column, e.g. `sample_name`, `tubule_type`, `class_name`) — then build:
  - an interactive **scatter plot** (like the UMAP embedding — click a point to crop and
    view its protein/lipid region from the source whole-slide image, only available at the
    tubule level),
  - a seaborn **boxplot/violin plot** (with pairwise Welch's t-test + Cohen's d between
    adjacent x categories) from any of its columns, with an optional hue column,
  - a **"Predicted-age trend"** plot: pick any numeric column from the combined tubule +
    shape tables (identifiers, coordinates, and the model's own prediction/attention
    outputs are excluded), and see the feature's trend against the model's own continuous
    `pred_320` prediction — outlier-trimmed (IQR) on both axes, binned into 0.2-wide
    `pred_320` buckets and averaged, then plotted with a regression line and its Spearman
    correlation (shown for any p-value, not just significant ones).

  Also links to two reference documents at the top of the page: a **Feature dictionary**
  (what every column in the Features/Statistics pages means) and **AI hypotheses**
  (confident kidney-aging hypotheses developed from the NoRI predicted-age maps, Grad-CAM
  observations, pathologist review, and quantitative feature discussions).
- **Statistics** — pick a feature file, aggregation level, and optional filter (same as
  Features), then screen every numeric feature with the full test suite at once, each
  ranked by its own Benjamini-Hochberg FDR-corrected q-value:
  - *Differences across all ages* (one row per feature): Kruskal-Wallis, Welch's ANOVA
    (parametric, robust to unequal variance), Spearman's ρ and a linear regression (is
    there a monotonic/linear trend with age), and a random-intercept mixed-effects model
    (`feature ~ age`, animal as random effect — uses every tubule while accounting for
    which animal it came from; requires `statsmodels`, degrades gracefully if missing).
  - *Differences between age groups* (one row per feature × adjacent age transition):
    Welch's t-test + Mann-Whitney U as a parametric/non-parametric pairwise pair, and
    Cohen's d + Cliff's delta as their matching effect sizes — the same comparison the
    Features-page boxplot annotates, run exhaustively for every feature and transition.
    This section also has an age-transition effect-size heatmap (rows = features,
    columns = transitions, color/value = Cohen's d) showing every feature-by-transition
    effect at once, not just each feature's single strongest transition.

  Each section gets its own ranked bar chart, volcano plot (effect size vs.
  significance), and results table — each table has its own copy (tab-separated, pastes
  into a spreadsheet) and download (.csv) buttons; pick any tested feature to drill into
  its boxplot/violin plot by age.

## License

MIT — see [LICENSE](LICENSE).
