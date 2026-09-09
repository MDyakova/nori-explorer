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
  optional hue column, across every image's feature file in the task.

## License

MIT — see [LICENSE](LICENSE).
