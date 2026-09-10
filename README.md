# NoRI Explorer

A browser-based tool ([Dash](https://dash.plotly.com/)) for exploring the outputs of NoRI aging-prediction
models: UMAP embeddings, regression metrics, attention scores, per-tile heatmaps, and
whole-image prediction overlays.

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

## Expected data layout

The data root passed via `--base-dir` (or entered in the UI) must contain `outputs/` and,
for the whole-image overlay, `data/`:

```
<base_dir>/
├── outputs/
│   └── <group>/
│       └── <task>/
│           ├── umap/
│           │   └── <model_name>.csv        # columns: age, filename, sample_name,
│           │                                #          umap1, umap2, prediction, ...
│           ├── max_values.csv               # columns: protein, lipid
│           ├── heatmaps/
│           │   └── <model_name>/<age>/<filename>   # per-tile heatmap images
│           ├── models/                       # optional training-curve images, named
│           │   ├── epoch_loss_<model_name>.*        # e.g. loss_<model>.png
│           │   ├── mae_<model_name>.*
│           │   └── r2_<model_name>.*
│           └── features/                     # optional, for the "Features" page
│               ├── <image_name>.csv                 # tubule-level features
│               ├── <image_name>_nucleolus.csv        # per-nucleolus features
│               └── <image_name>_shape.csv            # per-tubule shape features
└── data/
    └── <group>/
        └── <class_name>/
            └── <sample>_<class_name>_...tif  # whole-slide source images
```

## Features

- **Explorer** — pick a group / task / model, view the UMAP scatter plot, regression
  metrics (MAE, R², classification report, confusion matrix) and attention-score
  boxplots by age; click a point to inspect its tile heatmap.
- **All model plots** — side-by-side UMAP, density, and prediction plots (plus training
  curves) for every model in a task, with a combined prediction-by-sample boxplot.
- **Whole-image overlay** — renders a selected whole-slide `.tif` with per-tile
  predictions overlaid as a heatmap.
- **Features** — pick one of the three feature-file kinds (tubule, nucleolus, shape) and
  an aggregation level (raw tubules, per-animal median/mean, or per-animal-and-tubule-type
  median), then build an interactive scatter plot (like the UMAP embedding — click a
  point to crop and view its protein/lipid region from the source whole-slide image, only
  available at the tubule level) or a seaborn boxplot/violin plot (with pairwise Welch's
  t-test + Cohen's d between adjacent x categories) from any of its columns, with an
  optional hue column, across every image's feature file in the task. Also has a
  "Predicted-age trend" plot: pick any numeric column from the combined main + shape
  tables (identifiers, coordinates, and the model's own prediction/attention outputs are
  excluded), and see the feature's trend against the model's own continuous `pred_320`
  prediction — outlier-trimmed (IQR)
  on both axes, binned into 0.2-wide `pred_320` buckets and averaged, then plotted with a
  regression line and its Spearman correlation (shown for any p-value, not just
  significant ones).
- **Statistics** — pick a feature file and aggregation level, then screen every numeric
  feature with the full test suite at once, each ranked by its own Benjamini-Hochberg
  FDR-corrected q-value:
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
