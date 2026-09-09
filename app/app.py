"""NoRI interactive regression explorer - browser (Dash) version.

Ported from files/interactive_tool_outputs_regression.ipynb. The data root
(the folder that directly contains `outputs/` and `data/`) is chosen from
the browser UI - see the "Data folder" input at the top of the page - and
defaults to the current working directory or the --base-dir CLI argument.

Run with:
    python app/app.py [--base-dir /path/to/data/root]
"""
import argparse
import base64
import io
import json
import os
import threading
import traceback
import warnings
import webbrowser
from urllib.parse import parse_qs, urlencode

import dash
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from dash import Input, Output, State, dcc, html
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.stats import f as f_distribution
from scipy.stats import gaussian_kde, kruskal, linregress, mannwhitneyu, spearmanr, ttest_ind
from sklearn.metrics import mean_absolute_error, r2_score
from tifffile import TiffFile

try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    smf = None
    STATSMODELS_AVAILABLE = False


# Mirrors NORI_REGION_COLORS in assets/plot_toolbar.js so a region's badge color here
# matches the outline drawn on the image for it.
REGION_COLORS = ['#0f766e', '#b45309', '#7c3aed', '#be123c', '#0369a1', '#15803d']

# Calibration coefficients converting raw protein/lipid pixel intensities to
# quantitative units for region distribution/mean analysis.
PROTEIN_CALIBRATION_K = 1.3643 * 1000 / 8192
LIPID_CALIBRATION_K = 1.0101 * 1000 / 8192


def is_valid_base_dir(base_dir):
    return bool(base_dir) and os.path.isdir(os.path.join(base_dir, 'outputs'))


# --- data loading -----------------------------------------------------

def list_groups(base_dir):
    return sorted(os.listdir(os.path.join(base_dir, 'outputs')))


def list_tasks(base_dir, group):
    return sorted(os.listdir(os.path.join(base_dir, 'outputs', group)))


def list_models(base_dir, group, task):
    model_dir = os.path.join(base_dir, 'outputs', group, task, 'umap')
    return sorted(f.replace('.csv', '') for f in os.listdir(model_dir))


def load_umap_df(base_dir, group, task, model_name):
    umap_df = pd.read_csv(os.path.join(base_dir, 'outputs', group, task, 'umap', f'{model_name}.csv'))

    if pd.api.types.is_float_dtype(umap_df['age']):
        umap_df['age'] = umap_df['age'].astype('Int64')

    class_mapping = {cls: idx for idx, cls in enumerate(umap_df['age'].unique())}
    umap_df['class_numeric'] = umap_df['age'].map(class_mapping)

    umap_df['heatmap_path'] = [
        os.path.join('outputs', group, task, 'heatmaps', model_name, str(age), filename)
        for age, filename in zip(umap_df['age'], umap_df['filename'])
    ]

    umap_df['pred_class'] = umap_df['prediction'].apply(
        lambda p: 9 if p < 15 else 18 if p < 19.5 else 21 if p < 23 else 25
    )
    return umap_df


def load_max_values(base_dir, group, task):
    max_values = pd.read_csv(os.path.join(base_dir, 'outputs', group, task, 'max_values.csv'))
    return float(max_values['protein'].max()), float(max_values['lipid'].max())


def channels_txt_path(base_dir, group, task):
    return os.path.join(base_dir, 'outputs', group, task, 'channels.txt')


def load_channel_names(base_dir, group, task):
    """Read outputs/<group>/<task>/channels.txt (lines like '0:protein') into an
    ordered {index: name} dict. Falls back to {0: 'protein', 1: 'lipid'} if missing
    or unparsable."""
    path = channels_txt_path(base_dir, group, task)
    channels = {}
    if os.path.isfile(path):
        # utf-8-sig quietly strips a leading BOM, which Notepad-saved "UTF-8" files on
        # Windows often have and which would otherwise break parsing of the first line.
        with open(path, encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                idx_str, name = line.split(':', 1)
                idx_str = idx_str.strip()
                if idx_str.lstrip('-').isdigit():
                    channels[int(idx_str)] = name.strip()
    return channels if channels else {0: 'protein', 1: 'lipid'}


def find_channel_index(channel_names, target_name, fallback_idx):
    for idx, name in channel_names.items():
        if name.lower() == target_name:
            return idx
    return fallback_idx if fallback_idx in channel_names else next(iter(channel_names))


def compute_metrics(base_dir, umap_df, group, task):
    max_p, max_l = load_max_values(base_dir, group, task)

    mae = mean_absolute_error(umap_df['age'], umap_df['prediction'])
    r2 = r2_score(umap_df['age'], umap_df['prediction'])
    return mae, r2, max_p, max_l


# --- features data helpers -----------------------------------------------

# Each image_name in outputs/<group>/<task>/features/ has up to three CSVs: the
# tubule-level table itself (<image_name>.csv), and two suffixed side tables.
FEATURE_KINDS = {
    'main': 'Tubule features (<image_name>.csv)',
    'nucleolus': 'Nucleolus features (<image_name>_nucleolus.csv)',
    'shape': 'Shape features (<image_name>_shape.csv)',
}
_FEATURE_SUFFIXES = {'main': '.csv', 'nucleolus': '_nucleolus.csv', 'shape': '_shape.csv'}

FEATURE_AGGREGATION_LEVELS = {
    'tubule': 'Tubules (not aggregated)',
    'animal_median': 'Animal level (median)',
    'animal_mean': 'Animal level (mean)',
    'animal_tubule_median': 'Animal-tubule_type level (median)',
}

# Per-row bounding-box/id columns lose their meaning once rows are aggregated across
# tubules (a median bounding box doesn't correspond to any real crop region), so they're
# dropped rather than averaged whenever an aggregation level other than 'tubule' is used.
FEATURE_AGG_DROPPED_COLS = {'contour_id', 'label_id', 'min_x', 'min_y', 'max_x', 'max_y', 'step', 'step_n'}


def features_dir_path(base_dir, group, task):
    return os.path.join(base_dir, 'outputs', group, task, 'features')


def list_feature_files(base_dir, group, task, kind):
    """[(image_name, path), ...] for feature files of `kind` ('main'/'nucleolus'/'shape')."""
    feat_dir = features_dir_path(base_dir, group, task)
    if not os.path.isdir(feat_dir):
        return []

    suffix = _FEATURE_SUFFIXES[kind]
    other_suffixes = [s for k, s in _FEATURE_SUFFIXES.items() if k != kind and s != '.csv']

    items = []
    for fname in sorted(os.listdir(feat_dir)):
        if not fname.endswith('.csv'):
            continue
        if kind == 'main':
            if any(fname.endswith(s) for s in other_suffixes):
                continue
            image_name = fname[:-len('.csv')]
        else:
            if not fname.endswith(suffix):
                continue
            image_name = fname[:-len(suffix)]
        items.append((image_name, os.path.join(feat_dir, fname)))
    return items


def list_feature_kinds_available(base_dir, group, task):
    return [kind for kind in FEATURE_KINDS if list_feature_files(base_dir, group, task, kind)]


def get_feature_columns(base_dir, group, task, kind, level='tubule'):
    """Column names for `kind`, read from just the first file (schema is shared). At an
    aggregated `level`, the bbox/id columns that get dropped by aggregate_feature_table
    are excluded so they can't be picked for a plot in the first place."""
    files = list_feature_files(base_dir, group, task, kind)
    if not files:
        return []
    try:
        columns = list(pd.read_csv(files[0][1], nrows=0).columns)
    except Exception:
        return []
    if 'file_name_save' in columns and 'sample_name' not in columns:
        columns.append('sample_name')
    if level != 'tubule':
        columns = [c for c in columns if c not in FEATURE_AGG_DROPPED_COLS]
    return columns


def aggregate_feature_table(df, level):
    """Collapse the tubule-level table to one row per animal, or per animal x tubule
    type, matching the project's preference for animal-level statistics over raw,
    pseudoreplicated tubule counts. Bounding-box/id columns are dropped (see
    FEATURE_AGG_DROPPED_COLS) rather than averaged."""
    if level == 'tubule' or df.empty:
        return df

    if level == 'animal_median':
        group_cols, agg_func = ['sample_name'], 'median'
    elif level == 'animal_mean':
        group_cols, agg_func = ['sample_name'], 'mean'
    elif level == 'animal_tubule_median':
        group_cols, agg_func = ['sample_name', 'tubule_type'], 'median'
    else:
        return df

    if not all(c in df.columns for c in group_cols):
        return df

    df = df.drop(columns=[c for c in FEATURE_AGG_DROPPED_COLS if c in df.columns])

    numeric_cols = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in df.columns if c not in group_cols and c not in numeric_cols]

    agg_dict = {c: agg_func for c in numeric_cols}
    agg_dict.update({c: 'first' for c in other_cols})
    if not agg_dict:
        return df[group_cols].drop_duplicates().reset_index(drop=True)

    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def load_feature_table(base_dir, group, task, kind):
    """Concatenate every image's feature file of `kind` into one dataframe, adding a
    `sample_name` (animal) column derived the same way the umap CSVs' sample_name is."""
    frames = []
    for image_name, path in list_feature_files(base_dir, group, task, kind):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'file_name_save' not in df.columns:
            df['file_name_save'] = image_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined['sample_name'] = combined['file_name_save'].astype(str).apply(lambda p: p.split('_MAP')[0])

    if 'class_name' in combined.columns:
        combined['class_name'] = combined['class_name'].astype(int)
        combined.sort_values(by=['class_name'], inplace=True)
        combined['class_name'] = combined['class_name'].astype(str)

    return combined


# Columns available for the "Predicted-age trend" plot: the main tubule table's own
# metrics plus the shape table's morphology metrics, joined on (file_name_save,
# contour_id == label_id), plus two ratios derived from the main table.
def list_pred_age_trend_features(base_dir, group, task, pred_col='pred_320'):
    """Every numeric column in the combined main+shape table (see
    load_main_shape_merged_table), except identifiers/coordinates and the model's own
    prediction/attention outputs — pred_col is the x-axis here, so it (and its sibling
    pred/attn columns) aren't offered as a y-axis feature."""
    df = load_main_shape_merged_table(base_dir, group, task)
    if df.empty:
        return []

    excluded = FEATURE_NON_METRIC_COLS | {
        pred_col, 'pred_320', 'pred_128',
        'protein_attn_320', 'lipid_attn_320', 'protein_attn_128', 'lipid_attn_128',
    }
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def list_tubule_types(base_dir, group, task):
    """Distinct tubule_type values, read from just the first main feature file (the set
    of tubule types is assumed consistent across images in a group/task)."""
    files = list_feature_files(base_dir, group, task, 'main')
    if not files:
        return []
    try:
        values = pd.read_csv(files[0][1], usecols=['tubule_type'])['tubule_type']
    except Exception:
        return []
    return sorted(values.dropna().unique().tolist())


def load_main_shape_merged_table(base_dir, group, task):
    """The main tubule-level table left-joined with the shape table on
    (file_name_save, contour_id == label_id) — the shape table has no bbox/id columns
    of its own to look up by otherwise, but shares this id with the main table's
    contour_id. Also derives bb_size_k and lumen_size_k (bb_size / body_size,
    lumen_size / body_size)."""
    main_df = load_feature_table(base_dir, group, task, 'main')
    if main_df.empty:
        return pd.DataFrame()

    shape_df = load_feature_table(base_dir, group, task, 'shape')
    if not shape_df.empty and 'contour_id' in main_df.columns and 'label_id' in shape_df.columns:
        shape_only_cols = [c for c in shape_df.columns if c not in main_df.columns and c != 'label_id']
        merged = main_df.merge(
            shape_df[['file_name_save', 'label_id', *shape_only_cols]],
            left_on=['file_name_save', 'contour_id'], right_on=['file_name_save', 'label_id'],
            how='left',
        )
    else:
        merged = main_df.copy()

    if {'bb_size', 'body_size'}.issubset(merged.columns):
        merged['bb_size_k'] = merged['bb_size'] / merged['body_size']
    if {'lumen_size', 'body_size'}.issubset(merged.columns):
        merged['lumen_size_k'] = merged['lumen_size'] / merged['body_size']

    return merged


# --- image helpers ------------------------------------------------------

def pil_to_data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def plot_with_toolbar(img, filename):
    """Wrap a static plot html.Img with hover copy/download icon buttons."""
    return html.Div(className='plot-wrap', children=[
        img,
        html.Div(className='plot-toolbar', children=[
            html.Button(
                '⎘', title='Copy image', className='plot-icon-btn plot-copy-btn',
                **{'data-filename': filename},
            ),
            html.Button(
                '⬇', title='Download image', className='plot-icon-btn plot-download-btn',
                **{'data-filename': filename},
            ),
        ]),
    ])


def table_with_toolbar(table, filename):
    """Wrap an html.Table with copy/download icon buttons — copy puts a tab-separated
    version on the clipboard (pastes cleanly into a spreadsheet), download saves a
    .csv file."""
    return html.Div(className='table-wrap', children=[
        html.Div(className='table-toolbar', children=[
            html.Button(
                '⎘', title='Copy table (tab-separated)', className='table-icon-btn table-copy-btn',
                **{'data-filename': filename},
            ),
            html.Button(
                '⬇', title='Download table (CSV)', className='table-icon-btn table-download-btn',
                **{'data-filename': filename},
            ),
        ]),
        html.Div(table, style={'overflowX': 'auto'}),
    ])


# --- figure building ------------------------------------------------------

def build_attention_boxplots(umap_df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    order = sorted(umap_df['age'].dropna().unique())

    sns.boxplot(
        data=umap_df, x='age', y='protein_attn', hue='age', order=order, hue_order=order,
        palette='viridis', dodge=False, legend=False, ax=axes[0],
    )
    axes[0].set_title('Protein attention by aging group')
    axes[0].set_xlabel('Age')
    axes[0].grid(True)

    sns.boxplot(
        data=umap_df, x='age', y='lipid_attn', hue='age', order=order, hue_order=order,
        palette='viridis', dodge=False, legend=False, ax=axes[1],
    )
    axes[1].set_title('Lipid attention by aging group')
    axes[1].set_xlabel('Age')
    axes[1].grid(True)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_static_umap_image(umap_df, model_name):
    fig, ax = plt.subplots(figsize=(5, 4.2))

    order = sorted(umap_df['age'].dropna().unique())
    sns.scatterplot(
        data=umap_df, x='umap1', y='umap2', hue='age', hue_order=order,
        palette='viridis', s=14, linewidth=0, ax=ax,
    )
    ax.set_title(model_name, fontsize=11)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.grid(alpha=0.3)
    ax.legend(title='Age', fontsize=8, title_fontsize=8, markerscale=1.2, loc='best')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_static_umap_kde_image(umap_df, model_name):
    fig, ax = plt.subplots(figsize=(5, 4.2))

    order = sorted(umap_df['age'].dropna().unique())
    sns.kdeplot(
        data=umap_df, x='umap1', y='umap2', hue='age', hue_order=order,
        palette='viridis', fill=False, common_norm=True, ax=ax,
    )
    ax.set_title(f'{model_name} (density)', fontsize=11)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_static_umap_prediction_image(umap_df, model_name):
    fig, ax = plt.subplots(figsize=(5, 4.2))

    norm = matplotlib.colors.Normalize(vmin=umap_df['prediction'].min(), vmax=umap_df['prediction'].max())
    sns.scatterplot(
        data=umap_df, x='umap1', y='umap2', hue='prediction',
        palette='viridis', hue_norm=norm, s=20, linewidth=0, ax=ax,
    )
    ax.legend_.remove()

    sm = matplotlib.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Predicted age')

    ax.set_title(f'{model_name} (predicted age)', fontsize=11)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.grid(alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_combined_prediction_boxplot(base_dir, group, task, model_names):
    frames = []
    for model_name in model_names:
        try:
            frames.append(load_umap_df(base_dir, group, task, model_name))
        except Exception:
            continue

    if not frames:
        return None

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.dropna(subset=['age']).sort_values(by='age', key=lambda s: s.astype(int))
    age_order = combined_df['age'].astype(int).drop_duplicates().astype(str).tolist()
    combined_df['age'] = combined_df['age'].astype(str)

    n_samples = combined_df['sample_name'].nunique()
    fig, ax = plt.subplots(figsize=(max(9, n_samples * 0.5), 6))

    sns.boxplot(
        data=combined_df, x='sample_name', y='prediction', hue='age', hue_order=age_order,
        showfliers=False, ax=ax,
    )

    ax.set_xlabel('Sample')
    ax.set_ylabel('Prediction')
    ax.set_title('Predictions by age and sample (all models combined)')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def overlap_coefficient(x1, x2, n_points=512):
    """KDE-based overlap coefficient (shared area) between two 1-D samples."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if len(x1) < 2 or len(x2) < 2 or np.std(x1) == 0 or np.std(x2) == 0:
        return np.nan
    try:
        kde1 = gaussian_kde(x1)
        kde2 = gaussian_kde(x2)
    except np.linalg.LinAlgError:
        return np.nan
    lo = min(x1.min(), x2.min())
    hi = max(x1.max(), x2.max())
    grid = np.linspace(lo, hi, n_points)
    return float(np.trapz(np.minimum(kde1(grid), kde2(grid)), grid))


def tile_overlap_by_animal(df, animal_col='sample_name', age_col='age', pred_col='prediction', age_pairs=None):
    """Pairwise tile-distribution overlap between every animal at age1 and every animal at age2.

    age_pairs defaults to consecutive age transitions found in df, rather than a
    hardcoded kidney-aging schedule, so this generalizes across groups/tasks.
    """
    if age_pairs is None:
        ages = sorted(df[age_col].dropna().unique())
        age_pairs = list(zip(ages[:-1], ages[1:]))

    results = []
    for age1, age2 in age_pairs:
        df1 = df[df[age_col] == age1]
        df2 = df[df[age_col] == age2]

        animals1 = df1[animal_col].dropna().unique()
        animals2 = df2[animal_col].dropna().unique()

        for animal1 in animals1:
            x1 = df1.loc[df1[animal_col] == animal1, pred_col].dropna().values

            for animal2 in animals2:
                x2 = df2.loc[df2[animal_col] == animal2, pred_col].dropna().values

                if len(x1) < 2 or len(x2) < 2:
                    continue

                ovl = overlap_coefficient(x1, x2)
                if np.isnan(ovl):
                    continue

                results.append({
                    'comparison': f'{age1}m vs {age2}m',
                    'young_animal': animal1,
                    'old_animal': animal2,
                    'n_young_tiles': len(x1),
                    'n_old_tiles': len(x2),
                    'overlap': ovl,
                    'separation': 1 - ovl,
                })

    return pd.DataFrame(results)


def bootstrap_median_ci(x, n_boot=10000, ci=95, seed=42):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)

    boot_medians = np.array([
        np.median(rng.choice(x, size=len(x), replace=True))
        for _ in range(n_boot)
    ])

    alpha = (100 - ci) / 2
    return (
        np.median(x),
        np.percentile(boot_medians, alpha),
        np.percentile(boot_medians, 100 - alpha),
    )


def build_animal_level_separation_plot(base_dir, group, task, model_names):
    """Animal-level distribution-separation plot: for each age transition, aggregate
    tile-level overlap per young animal (averaged across old animals), then bootstrap
    the median separation by resampling animals - keeping the animal, not the tile, as
    the unit of replication."""
    frames = []
    for model_name in model_names:
        try:
            frames.append(load_umap_df(base_dir, group, task, model_name))
        except Exception:
            continue

    if not frames:
        return None

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.dropna(subset=['age', 'sample_name', 'prediction'])
    if combined_df.empty:
        return None
    combined_df['age'] = combined_df['age'].astype(int)

    ages = sorted(combined_df['age'].unique())
    age_pairs = list(zip(ages[:-1], ages[1:]))
    if not age_pairs:
        return None
    comparison_order = [f'{age1}m vs {age2}m' for age1, age2 in age_pairs]

    tile_animal_pairs = tile_overlap_by_animal(combined_df, age_pairs=age_pairs)
    if tile_animal_pairs.empty:
        return None

    tile_animal_stats = (
        tile_animal_pairs
        .groupby(['comparison', 'young_animal'], as_index=False)
        .agg(overlap=('overlap', 'mean'), separation=('separation', 'mean'))
    )

    rows = []
    for comparison in comparison_order:
        values = tile_animal_stats.loc[tile_animal_stats['comparison'] == comparison, 'separation'].dropna()
        if values.empty:
            continue
        median, ci_low, ci_high = bootstrap_median_ci(values.values)
        rows.append({'comparison': comparison, 'median': median, 'ci_low': ci_low, 'ci_high': ci_high})

    if not rows:
        return None

    tile_ci_df = pd.DataFrame(rows)
    x = np.arange(len(tile_ci_df))
    yerr = np.vstack([
        tile_ci_df['median'] - tile_ci_df['ci_low'],
        tile_ci_df['ci_high'] - tile_ci_df['median'],
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x, tile_ci_df['median'], yerr=yerr, marker='o', markersize=8, capsize=5,
        linewidth=2, label='Animal level',
    )

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' vs ', ' → ').replace('m', '') for c in tile_ci_df['comparison']])
    ax.set_xlabel('Age transition (months)')
    ax.set_ylabel('Distribution separation (1 − OVL)')
    ax.set_title('Animal-level prediction distribution separation by age transition')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_animal_median_age_trend_plot(base_dir, group, task, model_names):
    """Median predicted vs. actual age, computed from one median-prediction value per
    animal per model (not pooled tiles), so each animal contributes equally regardless
    of tile count. A single overall median line is drawn across all animals/models, with
    the underlying per-animal points colored by model so model-level spread is visible."""
    frames = []
    for model_name in model_names:
        try:
            frame = load_umap_df(base_dir, group, task, model_name)
        except Exception:
            continue
        frame = frame.copy()
        frame['model_name'] = model_name
        frames.append(frame)

    if not frames:
        return None

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.dropna(subset=['age', 'sample_name', 'prediction'])
    if combined_df.empty:
        return None
    combined_df['age'] = combined_df['age'].astype(int)

    animal_medians = (
        combined_df
        .groupby(['sample_name', 'age', 'model_name'], as_index=False)['prediction']
        .median()
    )

    age_ticks = sorted(animal_medians['age'].unique())
    age_min, age_max = age_ticks[0], age_ticks[-1]

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.lineplot(
        data=animal_medians, x='age', y='prediction',
        estimator='median', errorbar=None, linewidth=1.5, color='#444444',
        marker=None, ax=ax, label='Median (all animals)', zorder=1,
    )

    sns.scatterplot(
        data=animal_medians, x='age', y='prediction', hue='model_name',
        s=40, linewidth=0, ax=ax, zorder=2,
    )

    ax.plot(
        [age_min, age_max], [age_min, age_max], '--', linewidth=2, color='black',
        label='Ideal prediction (age = age)', zorder=1,
    )

    ax.set_xlabel('Age (months)')
    ax.set_ylabel('Predicted age (months)')
    ax.set_xticks(age_ticks)
    ax.set_title('Median predicted age by animal (dots colored by model)')
    ax.grid(alpha=0.3)
    ax.legend(title='Model', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8, title_fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


# --- features plots -------------------------------------------------------

def round_if_float(value, ndigits=2):
    return np.round(value, ndigits) if isinstance(value, float) else value


def filter_feature_rows(df, cols):
    """Drop rows with missing values or the -1 sentinel (used across the feature CSVs
    for "not computed") in any of `cols`."""
    cols = [c for c in cols if c]
    if not cols:
        return df

    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= df[col].notna()
        if pd.api.types.is_numeric_dtype(df[col]):
            mask &= df[col] != -1
    return df[mask]


FEATURE_ROW_ID_COLS = ('contour_id', 'label_id')
FEATURE_BBOX_COLS = ('min_x', 'min_y', 'max_x', 'max_y')


def build_feature_scatter_figure(df, x_col, y_col, hue_col=None):
    """Interactive scatter (like the UMAP embedding plot) — each point's customdata
    carries its image name plus either its own bounding box (the main tubule table) or
    its row id (nucleolus/shape tables, resolved against the main table on click), so a
    click can look up and crop the source protein/lipid image."""
    id_col = next((c for c in FEATURE_ROW_ID_COLS if c in df.columns), None)
    has_bbox = all(c in df.columns for c in FEATURE_BBOX_COLS)

    extra_cols = list(FEATURE_BBOX_COLS) if has_bbox else ([id_col] if id_col else [])
    plot_df = filter_feature_rows(df, [x_col, y_col, hue_col, *extra_cols])
    if plot_df.empty:
        return None

    def customdata_for(sub_df):
        n = len(sub_df)
        return np.column_stack([
            sub_df['file_name_save'] if 'file_name_save' in sub_df.columns else [''] * n,
            sub_df[id_col] if id_col else [np.nan] * n,
            sub_df['min_x'] if has_bbox else [np.nan] * n,
            sub_df['min_y'] if has_bbox else [np.nan] * n,
            sub_df['max_x'] if has_bbox else [np.nan] * n,
            sub_df['max_y'] if has_bbox else [np.nan] * n,
        ])

    hover = (
        f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Image: %{{customdata[0]}}<extra></extra>'
    )

    fig = go.Figure()
    if hue_col:
        for hue_value, sub_df in plot_df.groupby(hue_col):
            fig.add_trace(go.Scatter(
                x=sub_df[x_col], y=sub_df[y_col], mode='markers', name=str(hue_value),
                marker=dict(size=7), customdata=customdata_for(sub_df),
                hovertemplate=f'{hue_col}: {hue_value}<br>{hover}',
            ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df[x_col], y=plot_df[y_col], mode='markers',
            marker=dict(size=7, color='#3b82f6'), customdata=customdata_for(plot_df),
            hovertemplate=hover,
        ))

    fig.update_layout(
        template='plotly_white',
        title=f'{y_col} vs {x_col}' + (f' by {hue_col}' if hue_col else ''),
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(family='-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        height=560,
        legend_title_text=hue_col or None,
    )
    return fig


def get_feature_bbox_lookup(base_dir, group, task):
    """(file_name_save, contour_id) -> (min_x, min_y, max_x, max_y), from the main
    tubule-level feature table. The nucleolus/shape tables don't carry bbox columns
    themselves but share this id (as `label_id`) with the main table's `contour_id`."""
    main_df = load_feature_table(base_dir, group, task, 'main')
    if main_df.empty:
        return {}
    required = {'file_name_save', 'contour_id', *FEATURE_BBOX_COLS}
    if not required.issubset(main_df.columns):
        return {}

    lookup = {}
    for row in main_df[list(required)].dropna().itertuples(index=False):
        lookup[(row.file_name_save, row.contour_id)] = (row.min_x, row.min_y, row.max_x, row.max_y)
    return lookup


def build_feature_tile_crop_image(base_dir, group, task, image_name, min_x, min_y, max_x, max_y, min_display_px=220):
    """Crop the whole-slide .tif for `image_name` to [min_y:max_y, min_x:max_x] and
    render it as a protein/lipid composite, upscaled if the crop is very small."""
    _, tif_path = resolve_whole_image_path(base_dir, group, image_name)
    if tif_path is None:
        raise FileNotFoundError(f'Whole-slide image not found for {image_name}')

    with TiffFile(tif_path) as tif:
        image = tif.asarray()

    h, w = image.shape[1], image.shape[2]
    min_x, min_y, max_x, max_y = (int(round(v)) for v in (min_x, min_y, max_x, max_y))
    min_x, max_x = max(0, min_x), min(w, max_x)
    min_y, max_y = max(0, min_y), min(h, max_y)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError('Empty crop region.')

    crop = image[:, min_y:max_y, min_x:max_x]

    channel_names = load_channel_names(base_dir, group, task)
    max_p, max_l = load_max_values(base_dir, group, task)
    channel_indices = sorted({
        find_channel_index(channel_names, 'protein', 0),
        find_channel_index(channel_names, 'lipid', 1),
    })

    def channel_norm_max(idx):
        name = channel_names.get(idx, '').lower()
        return channel_norm_max_value(crop, idx, name, NORM_MODE_GLOBAL, max_p, max_l)

    composite = compose_channels(crop, channel_indices, channel_names, channel_norm_max)
    pil_img = Image.fromarray((composite * 255).astype(np.uint8))

    if max(pil_img.width, pil_img.height) < min_display_px:
        scale = min_display_px / max(pil_img.width, pil_img.height, 1)
        pil_img = pil_img.resize(
            (max(1, round(pil_img.width * scale)), max(1, round(pil_img.height * scale))),
            resample=Image.LANCZOS,
        )

    return pil_img, tif_path, (min_x, min_y, max_x, max_y)


def cohens_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_var = ((n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1)) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return np.nan
    return (x1.mean() - x2.mean()) / pooled_sd


def significance_stars(p_value):
    if np.isnan(p_value):
        return 'n/a'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def ordered_categories(values):
    """Sort unique category values numerically when possible (so e.g. age groups like
    '9'/'18'/'21'/'25' come out in age order, not string order), falling back to a plain
    sort otherwise."""
    unique_values = pd.unique(values)
    try:
        return sorted(unique_values, key=float)
    except (TypeError, ValueError):
        return sorted(unique_values, key=str)


def annotate_pairwise_stats(ax, plot_df, x_col, y_col, order):
    """Welch's t-test + Cohen's d between each pair of adjacent x categories (e.g.
    consecutive age groups), annotated as a bracket above the boxes/violins — a quick,
    honest read of how strong (and how significant) each step's difference is, without
    running every pair (which would multiply-test and clutter the plot)."""
    if not pd.api.types.is_numeric_dtype(plot_df[y_col]) or len(order) < 2:
        return

    groups = {cat: plot_df.loc[plot_df[x_col] == cat, y_col].dropna().values for cat in order}

    y_max = plot_df[y_col].max()
    y_min = plot_df[y_col].min()
    y_range = (y_max - y_min) or abs(y_max) or 1.0
    step = y_range * 0.08
    base = y_max + step

    for i in range(len(order) - 1):
        x1, x2 = groups[order[i]], groups[order[i + 1]]
        if len(x1) < 2 or len(x2) < 2:
            continue

        _, p_value = ttest_ind(x1, x2, equal_var=False)
        d = cohens_d(x1, x2)
        stars = significance_stars(p_value)
        d_text = f'd={d:.2f}' if not np.isnan(d) else 'd=n/a'

        y_bracket = base + i * step
        ax.plot(
            [i, i, i + 1, i + 1],
            [y_bracket, y_bracket + step * 0.15, y_bracket + step * 0.15, y_bracket],
            color='black', linewidth=1, clip_on=False,
        )
        ax.text(
            i + 0.5, y_bracket + step * 0.2, f'{d_text}, {stars}',
            ha='center', va='bottom', fontsize=8, clip_on=False,
        )

    ax.set_ylim(top=base + (len(order) - 1) * step + step * 1.5)


def build_feature_box_or_violin_plot(df, x_col, y_col, hue_col=None, plot_type='box'):
    plot_df = filter_feature_rows(df, [x_col, y_col, hue_col])
    if plot_df.empty:
        return None

    order = ordered_categories(plot_df[x_col])
    n_categories = max(len(order), 1)
    fig, ax = plt.subplots(figsize=(max(7, n_categories * 0.6), 6))
    if plot_type == 'violin':
        sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, order=order, cut=0, ax=ax)
    else:
        sns.boxplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, order=order, showfliers=False, ax=ax)

    # Pairwise stats between adjacent x categories only make sense to draw when there's
    # no hue splitting each category into sub-boxes.
    if hue_col is None:
        annotate_pairwise_stats(ax, plot_df, x_col, y_col, order)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plot_label = 'Violin plot' if plot_type == 'violin' else 'Boxplot'
    title = f'{plot_label}: {y_col} by {x_col}'
    ax.set_title(f'{title}, hue={hue_col}' if hue_col else title)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.grid(True, axis='y', alpha=0.3)
    if hue_col:
        ax.legend(title=hue_col, bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8, title_fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


TUBULE_TYPE_ALL = '__all__'


def build_pred_age_trend_plot(df, feature_col, pred_col='pred_320', tubule_type=TUBULE_TYPE_ALL, bin_width=0.2):
    """Predicted-age trend for one feature, optionally restricted to one tubule type
    (tubule_type=TUBULE_TYPE_ALL keeps every tubule type): trim outliers (IQR rule) on
    both the prediction and the feature, bin the prediction into `bin_width`-wide
    buckets and average within each bucket, then fit+plot a regression line and report
    the Spearman correlation between the binned prediction and the averaged feature
    value. Returns (image_data_uri, rho, p_value), or (None, None, None) if there isn't
    enough data to plot."""
    if not {pred_col, feature_col}.issubset(df.columns):
        return None, None, None

    mask = (df[pred_col] >= 5) & (df[pred_col] < 28) & (df[pred_col] != -1)
    if tubule_type and tubule_type != TUBULE_TYPE_ALL:
        if 'tubule_type' not in df.columns:
            return None, None, None
        mask &= df['tubule_type'] == tubule_type

    a = df[mask]
    a = filter_feature_rows(a, [pred_col, feature_col])
    if len(a) < 4:
        return None, None, None

    lower_p, upper_p = _iqr_bounds(a[pred_col].values)
    a = a[(a[pred_col] >= lower_p) & (a[pred_col] <= upper_p)]

    lower_c, upper_c = _iqr_bounds(a[feature_col].values)
    a = a[(a[feature_col] >= lower_c) & (a[feature_col] <= upper_c)]
    if len(a) < 4:
        return None, None, None

    binned = a[[pred_col, feature_col]].copy()
    binned[pred_col] = (binned[pred_col] // bin_width) * bin_width
    binned = binned.groupby(by=pred_col, as_index=False).mean()
    if len(binned) < 3:
        return None, None, None

    rho, p_value = spearmanr(binned[pred_col], binned[feature_col], nan_policy='omit')

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(
        data=binned, x=pred_col, y=feature_col,
        scatter_kws={'s': 15}, line_kws={'color': 'red'}, ax=ax,
    )
    ax.set_title(f'{feature_col}\nSpearman R={rho:.3f}, p={p_value:.3g}')
    ax.set_xlabel('Predicted age')
    ax.grid(True)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}', rho, p_value


# --- statistics page -------------------------------------------------------

# Columns that are identifiers/coordinates, not biological features, and so are
# excluded when scanning a feature table for "does this differ with age" candidates.
FEATURE_NON_METRIC_COLS = {'class_name', 'file_name_save', 'sample_name', 'tubule_type', 'nucleolus'} | FEATURE_AGG_DROPPED_COLS


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted q-values, same order as the input p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    result = np.empty(n)
    result[order] = q
    return result


def welch_anova(groups):
    """One-way ANOVA robust to unequal variances between groups (Welch, 1951).
    groups: a list of 1-D arrays, one per age group. Returns (F, p_value)."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    k = len(groups)
    if k < 2:
        return np.nan, np.nan

    n = np.array([len(g) for g in groups], dtype=float)
    mean = np.array([g.mean() for g in groups])
    var = np.array([g.var(ddof=1) for g in groups])
    var = np.where(var == 0, 1e-12, var)

    w = n / var
    w_sum = w.sum()
    grand_mean = np.sum(w * mean) / w_sum

    numerator = np.sum(w * (mean - grand_mean) ** 2) / (k - 1)
    lambda_term = np.sum((1 - w / w_sum) ** 2 / (n - 1))
    denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lambda_term
    if denominator <= 0:
        return np.nan, np.nan

    f_stat = numerator / denominator
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * lambda_term) if lambda_term > 0 else np.inf
    p_value = f_distribution.sf(f_stat, df1, df2)
    return f_stat, p_value


def cliffs_delta(x1, x2):
    """Non-parametric effect size: (#(x1>x2) - #(x1<x2)) / (n1*n2), computed via the
    Mann-Whitney U statistic (delta = 2*U/(n1*n2) - 1) so it stays fast even for large
    samples. Same sign convention as cohens_d: positive means x1 (younger) > x2 (older),
    i.e. the feature decreases with age."""
    n1, n2 = len(x1), len(x2)
    if n1 < 1 or n2 < 1:
        return np.nan
    try:
        u_stat, _ = mannwhitneyu(x1, x2, alternative='two-sided')
    except ValueError:
        return np.nan
    return (2 * u_stat) / (n1 * n2) - 1


def fit_mixed_effects_age(df, age_col, feature_col, group_col='sample_name'):
    """Random-intercept mixed model: feature ~ age, with animal (sample_name) as a
    random effect — lets every tubule contribute while still accounting for which
    animal it came from, rather than treating tubules as independent replicates.
    Convergence can fail with few animals or few tubules per animal (a known risk
    noted in the project's own statistics guidance); reported as a note, not raised."""
    if not STATSMODELS_AVAILABLE:
        return np.nan, np.nan, 'statsmodels not installed'
    if group_col not in df.columns:
        return np.nan, np.nan, 'no animal column'

    model_df = df[[age_col, feature_col, group_col]].dropna().copy()
    model_df['age_numeric'] = pd.to_numeric(model_df[age_col], errors='coerce')
    model_df = model_df.dropna(subset=['age_numeric', feature_col])
    if model_df[group_col].nunique() < 2 or len(model_df) < 4:
        return np.nan, np.nan, 'insufficient animals/rows'

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = smf.mixedlm(f'Q("{feature_col}") ~ age_numeric', model_df, groups=model_df[group_col])
            fit = model.fit(reml=False, method='lbfgs')
        coef = float(fit.params.get('age_numeric', np.nan))
        p_value = float(fit.pvalues.get('age_numeric', np.nan))
        note = 'ok' if fit.converged else 'did not converge'
        return coef, p_value, note
    except Exception:
        return np.nan, np.nan, 'failed to fit'


def compute_overall_feature_stats(df, age_col='class_name'):
    """For every numeric feature column, run the full suite of "does this differ across
    ALL ages" tests: Kruskal-Wallis (non-parametric omnibus), Welch's ANOVA (parametric,
    robust to unequal variance), Spearman's rho and a simple linear regression (is there
    a monotonic/linear trend with age), and a random-intercept mixed-effects model
    (feature ~ age, animal as random effect) so every tubule can be used while still
    accounting for which animal it came from. Each test's p-values get their own
    Benjamini-Hochberg q-value across every feature tested here; features are ranked by
    min(Kruskal-Wallis q, Welch's ANOVA q)."""
    if age_col not in df.columns:
        return pd.DataFrame()

    age_order = ordered_categories(df[age_col])
    feature_cols = [
        c for c in df.columns
        if c not in FEATURE_NON_METRIC_COLS and c != age_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    has_animal = 'sample_name' in df.columns

    rows = []
    for col in feature_cols:
        valid = filter_feature_rows(df, [age_col, col])
        if valid.empty:
            continue

        groups = [valid.loc[valid[age_col] == age, col].values for age in age_order]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        try:
            kw_stat, kw_p = kruskal(*groups)
        except ValueError:
            kw_stat, kw_p = np.nan, np.nan

        welch_f, welch_p = welch_anova(groups)

        age_numeric = pd.to_numeric(valid[age_col], errors='coerce')
        feature_values = valid[col].astype(float)
        trend_mask = age_numeric.notna()
        if trend_mask.sum() >= 3 and age_numeric[trend_mask].nunique() >= 2:
            rho, spearman_p = spearmanr(age_numeric[trend_mask], feature_values[trend_mask])
            lin = linregress(age_numeric[trend_mask], feature_values[trend_mask])
            slope, r_value, lin_p = lin.slope, lin.rvalue, lin.pvalue
        else:
            rho = spearman_p = slope = r_value = lin_p = np.nan

        if has_animal:
            mixed_coef, mixed_p, mixed_note = fit_mixed_effects_age(valid, age_col, col)
        else:
            mixed_coef, mixed_p, mixed_note = np.nan, np.nan, 'no animal column'

        rows.append({
            'feature': col,
            'n_groups': len(groups),
            'n_total': sum(len(g) for g in groups),
            'kw_stat': kw_stat, 'kw_p': kw_p,
            'welch_f': welch_f, 'welch_p': welch_p,
            'spearman_rho': rho, 'spearman_p': spearman_p,
            'linreg_slope': slope, 'linreg_r': r_value, 'linreg_p': lin_p,
            'mixed_coef': mixed_coef, 'mixed_p': mixed_p, 'mixed_note': mixed_note,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result['kw_q'] = benjamini_hochberg(result['kw_p'].fillna(1).values)
    result['welch_q'] = benjamini_hochberg(result['welch_p'].fillna(1).values)
    result['spearman_q'] = benjamini_hochberg(result['spearman_p'].fillna(1).values)
    result['linreg_q'] = benjamini_hochberg(result['linreg_p'].fillna(1).values)
    result['mixed_q'] = (
        benjamini_hochberg(result['mixed_p'].fillna(1).values) if result['mixed_p'].notna().any() else np.nan
    )

    result['q_value'] = result[['kw_q', 'welch_q']].min(axis=1)
    result['abs_spearman_rho'] = result['spearman_rho'].abs()
    result['direction'] = np.select(
        [result['spearman_rho'] > 0, result['spearman_rho'] < 0],
        ['increases with age', 'decreases with age'],
        default='n/a',
    )
    result['transition'] = None

    result.sort_values(by='q_value', inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def compute_pairwise_feature_stats(df, age_col='class_name'):
    """For every numeric feature column and every pair of adjacent age groups: Welch's
    t-test (parametric) and Mann-Whitney U (non-parametric) as a matched pair of
    significance tests, plus Cohen's d and Cliff's delta as their matching
    parametric/non-parametric effect sizes — the same adjacent-transitions convention
    the Features-page boxplot annotates, run exhaustively across every feature.
    FDR-corrected separately for each p-value column, across every (feature,
    transition) row; ranked by the Welch's t-test q-value."""
    if age_col not in df.columns:
        return pd.DataFrame()

    age_order = ordered_categories(df[age_col])
    if len(age_order) < 2:
        return pd.DataFrame()

    feature_cols = [
        c for c in df.columns
        if c not in FEATURE_NON_METRIC_COLS and c != age_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for col in feature_cols:
        valid = filter_feature_rows(df, [age_col, col])
        groups = {age: valid.loc[valid[age_col] == age, col].values for age in age_order}

        for i in range(len(age_order) - 1):
            a1, a2 = age_order[i], age_order[i + 1]
            x1, x2 = groups[a1], groups[a2]
            if len(x1) < 2 or len(x2) < 2:
                continue

            _, welch_p = ttest_ind(x1, x2, equal_var=False)
            try:
                _, mw_p = mannwhitneyu(x1, x2, alternative='two-sided')
            except ValueError:
                mw_p = np.nan
            if np.isnan(welch_p) and np.isnan(mw_p):
                continue

            d = cohens_d(x1, x2)
            delta = cliffs_delta(x1, x2)
            if np.isnan(d):
                direction = 'n/a'
            else:
                direction = 'increases with age' if d < 0 else 'decreases with age'

            rows.append({
                'feature': col,
                'transition': f'{a1} vs {a2}',
                'n1': len(x1),
                'n2': len(x2),
                'n_total': len(x1) + len(x2),
                'p_value': welch_p,
                'mannwhitney_p': mw_p,
                'cohens_d': d,
                'max_abs_cohens_d': abs(d) if not np.isnan(d) else np.nan,
                'cliffs_delta': delta,
                'direction': direction,
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result['q_value'] = benjamini_hochberg(result['p_value'].fillna(1).values)
    result['mannwhitney_q'] = benjamini_hochberg(result['mannwhitney_p'].fillna(1).values)
    result.sort_values(by='q_value', inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def build_stats_ranked_bar_plot(
    results_df, q_col='q_value', direction_col='direction',
    transition_col='transition', title_suffix='features differing across ages', top_n=20,
):
    if results_df.empty:
        return None

    top = results_df.head(top_n).iloc[::-1]
    neglog_q = -np.log10(top[q_col].clip(lower=1e-300))
    colors = ['#dc2626' if d == 'decreases with age' else '#2563eb' for d in top[direction_col]]
    if transition_col and transition_col in top.columns:
        labels = [f'{f} ({t})' if t else f for f, t in zip(top['feature'], top[transition_col])]
    else:
        labels = list(top['feature'])

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    ax.barh(labels, neglog_q, color=colors)
    ax.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('-log10(q-value)')
    ax.set_title(f'Top {len(top)} {title_suffix}')
    ax.legend(handles=[
        matplotlib.patches.Patch(color='#2563eb', label='increases with age'),
        matplotlib.patches.Patch(color='#dc2626', label='decreases with age'),
    ], loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_stats_volcano_plot(
    results_df, q_col='q_value', effect_col='max_abs_cohens_d', transition_col='transition',
    xlabel="Effect size (|Cohen's d|)", title='Feature significance vs. effect size (age differences)',
):
    if results_df.empty:
        return None

    x = results_df[effect_col]
    y = -np.log10(results_df[q_col].clip(lower=1e-300))
    significant = results_df[q_col] < 0.05
    has_transition = transition_col and transition_col in results_df.columns

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~significant], y[~significant], color='#9ca3af', s=25, alpha=0.7, label='q ≥ 0.05')
    ax.scatter(x[significant], y[significant], color='#dc2626', s=30, label='q < 0.05')

    for _, row in results_df[significant].nlargest(10, effect_col).iterrows():
        transition = row[transition_col] if has_transition else None
        label = f"{row['feature']} ({transition})" if transition else row['feature']
        ax.annotate(
            label, (row[effect_col], -np.log10(max(row[q_col], 1e-300))),
            fontsize=7, xytext=(4, 2), textcoords='offset points',
        )

    ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('-log10(q-value)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _transition_start_age(transition):
    try:
        return float(str(transition).split(' vs ')[0])
    except (ValueError, IndexError):
        return float('inf')


def build_pairwise_effect_size_heatmap(results_df, top_n=50):
    """Rows = features, columns = age transitions, cells = signed Cohen's d — the
    single view the ranked bar/volcano plots can't give you, since those only surface
    each feature's single strongest transition. Here every feature-by-transition effect
    size is visible at once, so you can see whether a feature's change is concentrated
    in one transition or spread across several, and compare directions across features."""
    if results_df.empty:
        return None

    transition_order = sorted(results_df['transition'].dropna().unique(), key=_transition_start_age)
    if not transition_order:
        return None

    # Same ranking convention as the ranked bar/volcano plots and results table: most
    # significant features first (by their best q-value across any transition).
    feature_rank = results_df.groupby('feature')['q_value'].min().sort_values()
    top_features = feature_rank.head(top_n).index.tolist()

    pivot = (
        results_df[results_df['feature'].isin(top_features)]
        .pivot(index='feature', columns='transition', values='cohens_d')
        .reindex(index=top_features, columns=transition_order)
    )
    if pivot.empty:
        return None

    values = pivot.values.astype(float)
    finite = values[np.isfinite(values)]
    max_abs = float(np.abs(finite).max()) if finite.size else 1.0
    max_abs = max_abs if max_abs > 0 else 1.0

    fig, ax = plt.subplots(
        figsize=(max(6, len(transition_order) * 1.7), max(4, len(top_features) * 0.32)),
    )
    im = ax.imshow(values, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='auto')

    ax.set_xticks(range(len(transition_order)))
    ax.set_xticklabels(transition_order, rotation=30, ha='right')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j, i, f'{value:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(value) > max_abs * 0.6 else 'black',
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d  (+ decreases with age, − increases with age)")
    ax.set_title(f'Age-transition effect-size heatmap (top {len(top_features)} features by q-value)')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _fmt_stat(value, spec='.3g'):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    if isinstance(value, str):
        return value
    return f'{value:{spec}}'


def build_overall_stats_table(results_df, max_rows=100):
    """Kruskal-Wallis, Welch's ANOVA, Spearman's rho, linear regression, and the
    mixed-effects model — one row per feature."""
    if results_df.empty:
        return html.Div('No features could be tested.', className='status-error')

    header = html.Tr([html.Th(h) for h in [
        'Feature', 'n (groups/total)',
        'Kruskal-Wallis H', 'KW p', 'KW q',
        "Welch's ANOVA F", 'Welch p', 'Welch q',
        'Spearman ρ', 'Spearman p', 'Spearman q',
        'Linear slope', 'Linear r', 'Linear p', 'Linear q',
        'Mixed-model coef (age)', 'Mixed p', 'Mixed q', 'Mixed note',
        'Direction',
    ]])
    body_rows = []
    for _, r in results_df.head(max_rows).iterrows():
        body_rows.append(html.Tr([
            html.Td(r['feature']),
            html.Td(f"{r['n_groups']} / {r['n_total']}"),
            html.Td(_fmt_stat(r['kw_stat'])), html.Td(_fmt_stat(r['kw_p'], '.2e')), html.Td(_fmt_stat(r['kw_q'], '.2e')),
            html.Td(_fmt_stat(r['welch_f'])), html.Td(_fmt_stat(r['welch_p'], '.2e')), html.Td(_fmt_stat(r['welch_q'], '.2e')),
            html.Td(_fmt_stat(r['spearman_rho'])), html.Td(_fmt_stat(r['spearman_p'], '.2e')), html.Td(_fmt_stat(r['spearman_q'], '.2e')),
            html.Td(_fmt_stat(r['linreg_slope'])), html.Td(_fmt_stat(r['linreg_r'])), html.Td(_fmt_stat(r['linreg_p'], '.2e')), html.Td(_fmt_stat(r['linreg_q'], '.2e')),
            html.Td(_fmt_stat(r['mixed_coef'])), html.Td(_fmt_stat(r['mixed_p'], '.2e')), html.Td(_fmt_stat(r['mixed_q'], '.2e')), html.Td(_fmt_stat(r['mixed_note'])),
            html.Td(r['direction']),
        ]))

    note = None
    if len(results_df) > max_rows:
        note = html.P(
            f"Showing the top {max_rows} of {len(results_df)} tested features (sorted by min(KW q, Welch's ANOVA q)).",
            className='status-text',
        )
    return html.Div([
        note,
        table_with_toolbar(
            html.Table(className='stats-table', children=[html.Thead(header), html.Tbody(body_rows)]),
            'overall_age_stats',
        ),
    ])


def build_pairwise_stats_table(results_df, max_rows=100):
    """Welch's t-test, Mann-Whitney U, Cohen's d, and Cliff's delta — one row per
    (feature, adjacent age transition)."""
    if results_df.empty:
        return html.Div('No features could be tested.', className='status-error')

    header = html.Tr([html.Th(h) for h in [
        'Feature', 'Transition', 'n (1/2)',
        "Welch's t p", 'Welch q', 'Mann-Whitney p', 'MW q',
        "Cohen's d", "Cliff's delta", 'Direction',
    ]])
    body_rows = []
    for _, r in results_df.head(max_rows).iterrows():
        body_rows.append(html.Tr([
            html.Td(r['feature']),
            html.Td(r['transition']),
            html.Td(f"{r['n1']} / {r['n2']}"),
            html.Td(_fmt_stat(r['p_value'], '.2e')), html.Td(_fmt_stat(r['q_value'], '.2e')),
            html.Td(_fmt_stat(r['mannwhitney_p'], '.2e')), html.Td(_fmt_stat(r['mannwhitney_q'], '.2e')),
            html.Td(_fmt_stat(r['cohens_d'])), html.Td(_fmt_stat(r['cliffs_delta'])),
            html.Td(r['direction']),
        ]))

    note = None
    if len(results_df) > max_rows:
        note = html.P(
            f"Showing the top {max_rows} of {len(results_df)} tested rows (sorted by Welch's t-test q-value).",
            className='status-text',
        )
    return html.Div([
        note,
        table_with_toolbar(
            html.Table(className='stats-table', children=[html.Thead(header), html.Tbody(body_rows)]),
            'pairwise_age_stats',
        ),
    ])


# --- whole-image prediction overlay -------------------------------------

def image_filter(image):
    """Clip each channel at the median per-tile 99th percentile to remove outliers."""
    all_layers = []
    for layer in range(image.shape[0]):
        image_layer = image[layer]
        all_percentile = []
        for step_i in range(image_layer.shape[0] // 256):
            for step_j in range(image_layer.shape[1] // 256):
                crop = image_layer[step_i * 256:(step_i + 1) * 256, step_j * 256:(step_j + 1) * 256]
                all_percentile.append(np.percentile(crop, 99))
        percentile_99 = np.median(all_percentile) if all_percentile else np.percentile(image_layer, 99)
        image_layer = np.where(image_layer > percentile_99, percentile_99, image_layer)
        all_layers.append(image_layer)
    return np.array(all_layers)


def compute_whole_image_name(umap_df):
    return umap_df['filename'].apply(lambda p: '_'.join(p.split('_')[:-2]))


def list_whole_images(base_dir, group, task, model_name):
    umap_df = load_umap_df(base_dir, group, task, model_name)
    whole_image_name = compute_whole_image_name(umap_df)
    return sorted(whole_image_name.unique())


def list_data_tif_images(base_dir, group):
    """List every .tif/.tiff under base_dir/data/<group>/<class>/, as paths relative to that folder."""
    data_group_dir = os.path.join(base_dir, 'data', group)
    if not os.path.isdir(data_group_dir):
        return []

    images = []
    for class_name in sorted(os.listdir(data_group_dir)):
        class_dir = os.path.join(data_group_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.tif', '.tiff')):
                images.append(os.path.join(class_name, fname))
    return images


def add_tile_xy(umap_df_im):
    umap_df_im = umap_df_im.copy()
    umap_df_im['y'] = umap_df_im['filename'].apply(lambda p: int(p.split('.jpg')[0].split('_')[-2]))
    umap_df_im['x'] = umap_df_im['filename'].apply(lambda p: int(p.split('.jpg')[0].split('_')[-1]))
    return umap_df_im


def infer_tile_size(umap_df_im):
    """Tiles are laid out with a step of tile_size/2, so twice the smallest
    gap between consecutive x or y offsets recovers the tile size."""
    umap_df_im = add_tile_xy(umap_df_im)
    xs = sorted(umap_df_im['x'].unique())
    ys = sorted(umap_df_im['y'].unique())
    diffs = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0]
    diffs += [b - a for a, b in zip(ys, ys[1:]) if b - a > 0]
    return min(diffs) * 2 if diffs else None


def get_tiles_for_image(base_dir, group, task, model_name, image_name):
    umap_df = load_umap_df(base_dir, group, task, model_name)
    umap_df['whole_image_name'] = compute_whole_image_name(umap_df)
    return umap_df[umap_df['whole_image_name'] == image_name].copy()


def resolve_whole_image_path(base_dir, group, image_name):
    data_group_dir = os.path.join(base_dir, 'data', group)
    if not os.path.isdir(data_group_dir):
        return data_group_dir, None

    for class_name in sorted(os.listdir(data_group_dir)):
        if '_' + class_name in image_name:
            candidate = os.path.join(data_group_dir, class_name, image_name + '.tif')
            if os.path.exists(candidate):
                return data_group_dir, candidate

    return data_group_dir, None


ORIGINAL_OVERLAY_FIG_WIDTH = 30
ORIGINAL_OVERLAY_FIG_HEIGHT = 10
OVERLAY_PRED_VMIN = 8
OVERLAY_PRED_VMAX = 26


def build_whole_image_overlay(
    base_dir, group, task, model_name, image_name,
    fig_width=ORIGINAL_OVERLAY_FIG_WIDTH, fig_height=ORIGINAL_OVERLAY_FIG_HEIGHT,
):
    data_group_dir, tif_path = resolve_whole_image_path(base_dir, group, image_name)
    if tif_path is None:
        raise FileNotFoundError(
            f'No matching .tif for "{image_name}" found under {data_group_dir} '
            f'(looked in each class subfolder for a name containing "_<class>").'
        )

    max_p, max_l = load_max_values(base_dir, group, task)
    umap_df_im = add_tile_xy(get_tiles_for_image(base_dir, group, task, model_name, image_name))

    tile_size = infer_tile_size(umap_df_im)
    if tile_size is None:
        raise ValueError(
            f'Could not infer tile size for "{image_name}": fewer than 2 distinct tile '
            f'x/y offsets were found in the umap CSV.'
        )

    with TiffFile(tif_path) as tif:
        image = tif.asarray()

    image_nori = np.stack([image[0], image[1], image[0] * 0], axis=0).astype(float)
    filtered_image = image_filter(image_nori)
    filtered_image[0] = filtered_image[0] / max_p
    filtered_image[1] = filtered_image[1] / max_l
    filtered_image = np.clip(filtered_image, 0, 1)

    transformed_image = (filtered_image * 255).transpose((1, 2, 0)).astype(np.uint8)
    gray_image = (
        0.2989 * transformed_image[:, :, 0] +
        0.5870 * transformed_image[:, :, 1] +
        0.1140 * transformed_image[:, :, 2]
    )

    pred_map = np.zeros_like(image[0], dtype=float)
    count_map = np.zeros_like(image[0], dtype=float)

    for idx in umap_df_im.index:
        y = umap_df_im.loc[idx, 'y']
        x = umap_df_im.loc[idx, 'x']
        prediction = umap_df_im.loc[idx, 'prediction']
        pred_map[y:y + tile_size, x:x + tile_size] += prediction
        count_map[y:y + tile_size, x:x + tile_size] += 1

    pred_map = np.divide(pred_map, count_map, out=np.zeros_like(pred_map), where=count_map > 0)
    pred_overlay = np.ma.masked_where(count_map == 0, pred_map)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(gray_image, cmap='gray')
    ax.imshow(pred_overlay, cmap='jet', alpha=0.45, vmin=OVERLAY_PRED_VMIN, vmax=OVERLAY_PRED_VMAX)
    ax.axis('off')

    # No title/padding baked in: the saved PNG's content must line up pixel-for-pixel
    # (as a plain fraction-of-width/height) with the original image, so that a click's
    # fractional position on it can be mapped back to original-image tiles — see
    # find_tiles_at_fraction.
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}', tif_path, tile_size


def find_tiles_at_fraction(base_dir, group, task, model_name, image_name, frac_x, frac_y):
    """Given a click position as a fraction (0..1) of the whole-image overlay's
    displayed width/height, return every tile (a row of `model_name`'s umap CSV, with
    its filename/age/x/y) whose tile_size x tile_size box contains that point, ordered
    top-to-bottom/left-to-right. Tiles are laid out on a grid with a step of
    tile_size // 2 in each direction (see infer_tile_size), so a given point is
    typically covered by up to 4 overlapping tiles — fewer only near the image edges."""
    _, tif_path = resolve_whole_image_path(base_dir, group, image_name)
    if tif_path is None:
        return []

    with TiffFile(tif_path) as tif:
        image = tif.asarray()
    height, width = image.shape[1], image.shape[2]
    click_x = frac_x * width
    click_y = frac_y * height

    umap_df_im = add_tile_xy(get_tiles_for_image(base_dir, group, task, model_name, image_name))
    tile_size = infer_tile_size(umap_df_im)
    if tile_size is None:
        return []

    matches = umap_df_im[
        (umap_df_im['x'] <= click_x) & (click_x < umap_df_im['x'] + tile_size) &
        (umap_df_im['y'] <= click_y) & (click_y < umap_df_im['y'] + tile_size)
    ].sort_values(['y', 'x'])

    return [
        {'filename': row['filename'], 'age': row['age'], 'x': int(row['x']), 'y': int(row['y'])}
        for _, row in matches.iterrows()
    ]


def collect_tile_heatmaps(base_dir, group, task, model_name, tiles):
    """The heatmap image path for each of the given tiles (in `model_name`'s heatmaps
    folder), skipping any tile whose heatmap file isn't actually on disk."""
    heatmaps = []
    for tile in tiles:
        rel_path = os.path.join(
            'outputs', group, task, 'heatmaps', model_name, str(tile['age']), tile['filename'],
        )
        if os.path.isfile(os.path.join(base_dir, rel_path)):
            heatmaps.append({
                'filename': tile['filename'], 'x': tile['x'], 'y': tile['y'], 'path': rel_path,
            })
    return heatmaps


def build_overlay_colorbar():
    """Standalone 'Prediction' colorbar for the whole-image overlay, matching the jet
    colormap/range used to render the overlay itself — rendered as its own image next
    to the main overlay image, rather than baked into it, so the zoom pixel coordinates
    on that image stay untouched (matches how build_channel_colorbars works for the
    Image viewer)."""
    fig, ax = plt.subplots(figsize=(0.7, 1.75), constrained_layout=True)
    norm = matplotlib.colors.Normalize(vmin=OVERLAY_PRED_VMIN, vmax=OVERLAY_PRED_VMAX)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='jet'), cax=ax)
    cbar.set_label('Prediction', fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# protein/lipid keep their familiar red/green look; any other selected channel cycles
# through this palette so any number of channels (1, 2, 3, ...) can be shown at once.
CHANNEL_DEFAULT_COLORS = {'protein': (1.0, 0.0, 0.0), 'lipid': (0.0, 1.0, 0.0)}
CHANNEL_COLOR_PALETTE = [
    (0.0, 0.45, 1.0),   # blue
    (1.0, 0.0, 1.0),    # magenta
    (1.0, 0.85, 0.0),   # yellow
    (0.0, 1.0, 1.0),    # cyan
    (1.0, 0.5, 0.0),    # orange
    (0.6, 0.0, 1.0),    # purple
]


def channel_color(name, palette_i):
    """Color assigned to a channel for display: protein=red, lipid=green, everything
    else cycles through CHANNEL_COLOR_PALETTE using palette_i (the count of prior
    non-protein/lipid channels seen so far, in selection order)."""
    key = name.lower()
    if key in CHANNEL_DEFAULT_COLORS:
        return CHANNEL_DEFAULT_COLORS[key]
    return CHANNEL_COLOR_PALETTE[palette_i % len(CHANNEL_COLOR_PALETTE)]


NORM_MODE_GLOBAL = 'global'
NORM_MODE_PER_IMAGE = 'per_image'
NORM_MODE_OPTIONS = [
    {
        'label': 'Protein/lipid: calibration file max (max_values.csv)',
        'value': NORM_MODE_GLOBAL,
    },
    {
        'label': "Every channel: this image's own max after filtering",
        'value': NORM_MODE_PER_IMAGE,
    },
]


def channel_norm_max_value(image, idx, name, norm_mode, max_p, max_l):
    """Normalization divisor for one channel's display intensity (composite image and
    its colorbar) — purely a display scale, independent of the calibrated protein/lipid
    values (mg/mL) computed elsewhere.
    - NORM_MODE_GLOBAL: protein/lipid are divided by the calibration file's max
      (max_values.csv, shared across every image in the task) so intensities are
      comparable image-to-image; any other channel is stretched to its own
      99.5th-percentile intensity in the raw image.
    - NORM_MODE_PER_IMAGE: every channel (including protein/lipid) is stretched to its
      own max value after image_filter's per-image outlier clipping, i.e. the same
      filtered layer compose_channels actually displays — so each image uses its own
      full display range regardless of the calibration file."""
    if norm_mode == NORM_MODE_PER_IMAGE:
        filtered = image_filter(image[idx:idx + 1].astype(float))[0]
        value = float(filtered.max())
        return value if value > 0 else 1.0
    if name == 'protein':
        return max_p
    if name == 'lipid':
        return max_l
    value = float(np.percentile(image[idx], 99.5))
    return value if value > 0 else 1.0


def compose_channels(image, channel_indices, channel_names, channel_norm_max):
    """Additively blend the given channel indices of a (C, H, W) raw image array into
    one (H, W, 3) float composite in [0, 1]. Each channel is tinted by its own color —
    protein=red, lipid=green by default, everything else cycles through a fixed
    palette — so any number of channels (1, 2, 3, ...) can be shown at once."""
    h, w = image.shape[1], image.shape[2]
    composite = np.zeros((h, w, 3), dtype=float)
    palette_i = 0
    for idx in channel_indices:
        name = channel_names.get(idx, str(idx)).lower()
        channel_layer = image_filter(image[idx:idx + 1].astype(float))[0]
        channel_layer = channel_layer / channel_norm_max(idx)

        color = channel_color(name, palette_i)
        if name not in CHANNEL_DEFAULT_COLORS:
            palette_i += 1

        for c in range(3):
            if color[c]:
                composite[:, :, c] += channel_layer * color[c]

    return np.clip(composite, 0, 1)


def build_plain_nori_image(base_dir, group, task, rel_image_path, scale=1.0, channel_indices=None,
                            norm_mode=NORM_MODE_GLOBAL):
    """Render a color composite for one whole-slide .tif, resized to `scale` of its
    original size. channel_indices (a list, e.g. [0], [0, 1], [0, 1, 2, 3]) picks which
    raw channels (see channels.txt) to show — each is tinted by its own color (protein
    red, lipid green by default; other channels cycle through a fixed palette) and
    additively blended into one RGB image, so any number of channels can be shown.
    norm_mode (NORM_MODE_GLOBAL or NORM_MODE_PER_IMAGE) picks how each channel's display
    intensity is normalized — see channel_norm_max_value. Also returns
    {channel_idx: normalization_max} so a caller can build matching colorbars without
    re-reading the tif or redoing the percentile computation."""
    tif_path = os.path.join(base_dir, 'data', group, rel_image_path)
    channel_names = load_channel_names(base_dir, group, task)
    max_p, max_l = load_max_values(base_dir, group, task)
    channel_indices = channel_indices or [find_channel_index(channel_names, 'protein', 0)]

    with TiffFile(tif_path) as tif:
        image = tif.asarray()

    channel_max_values = {}

    def channel_norm_max(idx):
        name = channel_names.get(idx, '').lower()
        value = channel_norm_max_value(image, idx, name, norm_mode, max_p, max_l)
        channel_max_values[idx] = value
        return value

    composite = compose_channels(image, channel_indices, channel_names, channel_norm_max)
    transformed_image = (composite * 255).astype(np.uint8)
    pil_img = Image.fromarray(transformed_image)

    width = max(1, round(pil_img.width * scale))
    height = max(1, round(pil_img.height * scale))
    if (width, height) != pil_img.size:
        pil_img = pil_img.resize((width, height), resample=Image.LANCZOS)

    return pil_img, tif_path, channel_max_values


def build_channel_colorbars(channel_indices, channel_names, channel_max_values):
    """One vertical colorbar per selected channel, matching that channel's tint color
    (protein=red, lipid=green, others per CHANNEL_COLOR_PALETTE) and 0..max range —
    rendered as its own standalone image next to the main viewer image, rather than
    baked into it, so the zoom/region-select pixel coordinates on that image stay
    untouched. Protein/lipid are labeled in mg/mL."""
    n = len(channel_indices)
    fig, axes = plt.subplots(1, n, figsize=(0.7 * n, 1.75), constrained_layout=True)
    axes = [axes] if n == 1 else list(axes)

    palette_i = 0
    for ax, idx in zip(axes, channel_indices):
        name = channel_names.get(idx, str(idx))
        key = name.lower()
        color = channel_color(name, palette_i)
        if key not in CHANNEL_DEFAULT_COLORS:
            palette_i += 1

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f'ch_{idx}', [(0, 0, 0), color])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=channel_max_values.get(idx, 1.0))
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
        label = f'{name} (mg/mL)' if key in ('protein', 'lipid') else name
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _iqr_bounds(values, k=1.5):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return -np.inf, np.inf
    return q1 - k * iqr, q3 + k * iqr


def remove_outliers(protein, lipid, k=1.5):
    """Drop pixels whose protein or lipid value is an outlier (outside
    Q1 - k*IQR, Q3 + k*IQR) for its own channel."""
    p_lo, p_hi = _iqr_bounds(protein, k)
    l_lo, l_hi = _iqr_bounds(lipid, k)
    mask = (protein >= p_lo) & (protein <= p_hi) & (lipid >= l_lo) & (lipid <= l_hi)
    if mask.sum() < 2:
        return protein, lipid
    return protein[mask], lipid[mask]


def remove_outliers_1d(values, k=1.5):
    """Drop values outside Q1 - k*IQR, Q3 + k*IQR for a single channel."""
    lo, hi = _iqr_bounds(values, k)
    mask = (values >= lo) & (values <= hi)
    return values[mask] if mask.sum() >= 2 else values


def region_bbox_and_mask(points, kind, w, h):
    """points: list of {'x': frac, 'y': frac} in [0, 1] (relative to the image's own
    width/height). Returns the pixel bounding box (x0, y0, x1, y1) plus a boolean mask
    over that box selecting the pixels actually inside the region — None for a
    rectangle (the whole crop is the region), or a polygon-shaped mask for a polygon."""
    xs = [p['x'] * w for p in points]
    ys = [p['y'] * h for p in points]
    x0 = int(np.clip(round(min(xs)), 0, w))
    x1 = int(np.clip(round(max(xs)), 0, w))
    y0 = int(np.clip(round(min(ys)), 0, h))
    y1 = int(np.clip(round(max(ys)), 0, h))

    if kind != 'polygon':
        return x0, y0, x1, y1, None

    poly_path = MplPath(list(zip(xs, ys)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sample_points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    mask = poly_path.contains_points(sample_points).reshape(yy.shape) if xx.size else np.zeros(xx.shape, dtype=bool)
    return x0, y0, x1, y1, mask


def build_region_analysis(base_dir, group, task, rel_image_path, region, max_thumb_px=220,
                           channel_indices=None, norm_mode=NORM_MODE_GLOBAL):
    """For a region — {'type': 'rect', 'points': [{x0,y0}, {x1,y1}]} or
    {'type': 'polygon', 'points': [{x,y}, ...]} (all fractional, relative to the image's
    own width/height) — return a composite thumbnail of the cropped region (pixels
    outside a polygon are blacked out) — showing every channel in channel_indices
    (default: protein/lipid), tinted/blended the same way as the main viewer (norm_mode
    picks the same display normalization — see channel_norm_max_value) — plus its
    (outlier-removed, calibrated) protein/lipid pixel values and means, which are always
    computed from raw pixel intensities and unaffected by norm_mode."""
    tif_path = os.path.join(base_dir, 'data', group, rel_image_path)
    channel_names = load_channel_names(base_dir, group, task)
    max_p, max_l = load_max_values(base_dir, group, task)
    protein_idx = find_channel_index(channel_names, 'protein', 0)
    lipid_idx = find_channel_index(channel_names, 'lipid', 1)
    channel_indices = channel_indices or [protein_idx, lipid_idx]

    with TiffFile(tif_path) as tif:
        image = tif.asarray()

    h, w = image[0].shape
    x0, y0, x1, y1, mask = region_bbox_and_mask(region['points'], region.get('type', 'rect'), w, h)
    if x1 - x0 < 2 or y1 - y0 < 2 or (mask is not None and mask.sum() < 4):
        raise ValueError('Selected region is too small to analyze.')

    def channel_norm_max(idx):
        name = channel_names.get(idx, '').lower()
        return channel_norm_max_value(image, idx, name, norm_mode, max_p, max_l)

    image_crop = image[:, y0:y1, x0:x1]
    if mask is not None:
        image_crop = image_crop.copy()
        image_crop[:, ~mask] = 0
    composite = compose_channels(image_crop, channel_indices, channel_names, channel_norm_max)
    thumb = Image.fromarray((composite * 255).astype(np.uint8))
    thumb_scale = min(1.0, max_thumb_px / max(thumb.width, thumb.height))
    if thumb_scale < 1.0:
        thumb = thumb.resize(
            (max(1, round(thumb.width * thumb_scale)), max(1, round(thumb.height * thumb_scale))),
            resample=Image.LANCZOS,
        )

    def masked_values(idx):
        crop = image[idx][y0:y1, x0:x1].astype(float)
        return crop[mask] if mask is not None else crop.ravel()

    protein_raw = masked_values(protein_idx) * PROTEIN_CALIBRATION_K
    lipid_raw = masked_values(lipid_idx) * LIPID_CALIBRATION_K
    protein, lipid = remove_outliers(protein_raw, lipid_raw)
    n_outliers = protein_raw.size - protein.size
    mean_protein = float(protein.mean())
    mean_lipid = float(lipid.mean())

    # Any other selected channel (e.g. AQP2) also gets its own outlier-removed pixel
    # values, in selection order, so it can show up in the cross-region boxplot too —
    # kept separate from protein/lipid since those have real calibration/units and
    # drive the joint KDE plot, while these are just raw per-channel intensities.
    extra_channels = []
    for idx in channel_indices:
        if idx in (protein_idx, lipid_idx):
            continue
        name = channel_names.get(idx, str(idx))
        values = remove_outliers_1d(masked_values(idx))
        extra_channels.append((name, values))

    return thumb, (x0, y0, x1, y1), protein, lipid, mean_protein, mean_lipid, n_outliers, extra_channels


def build_combined_distribution_plot(region_results, max_points_per_region=5000):
    """Overlay every selected region's protein-vs-lipid pixel distribution on one
    large kdeplot, colored and labeled by region so regions can be compared directly."""
    rng = np.random.default_rng(0)
    frames = []
    for i, r in enumerate(region_results):
        protein, lipid = r['protein'], r['lipid']
        if protein.size > max_points_per_region:
            idx = rng.choice(protein.size, max_points_per_region, replace=False)
            protein, lipid = protein[idx], lipid[idx]
        frames.append(pd.DataFrame({'protein': protein, 'lipid': lipid, 'region': f'Region {i + 1}'}))
    combined_df = pd.concat(frames, ignore_index=True)

    palette = {f'Region {i + 1}': REGION_COLORS[i % len(REGION_COLORS)] for i in range(len(region_results))}

    fig, ax = plt.subplots(figsize=(4.0625, 3.75))
    try:
        sns.kdeplot(
            data=combined_df, x='protein', y='lipid', hue='region', palette=palette,
            levels=6, thresh=0.05, linewidths=1.6, ax=ax,
        )
    except Exception:
        sns.scatterplot(
            data=combined_df, x='protein', y='lipid', hue='region', palette=palette,
            s=8, alpha=0.35, linewidth=0, ax=ax,
        )
    if ax.get_legend() is not None:
        sns.move_legend(ax, 'center left', bbox_to_anchor=(1.02, 0.5), frameon=True, title=None)
    ax.set_xlabel('Protein (mg/mL)')
    ax.set_ylabel('Lipid (mg/mL)')
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    subsampled = any(r['protein'].size > max_points_per_region for r in region_results)
    subtitle = f'{len(region_results)} region(s)'
    if subsampled:
        subtitle += f'  ·  each capped to {max_points_per_region} px for plotting'
    ax.set_title(subtitle, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def build_region_boxplot(region_results):
    """One boxplot subplot per selected channel (protein, lipid, plus any other channels
    that were selected in the viewer) — small multiples, since channels can sit on very
    different intensity scales and would otherwise squash each other on a shared axis.
    Each subplot groups that channel's outlier-removed values by region. Sized to match
    build_combined_distribution_plot's KDE plot, since the two are shown stacked."""
    channel_data = [
        ('Protein (mg/mL)', [r['protein'] for r in region_results]),
        ('Lipid (mg/mL)', [r['lipid'] for r in region_results]),
    ]
    extra_names = [name for name, _ in region_results[0]['extra_channels']]
    for i, name in enumerate(extra_names):
        channel_data.append((name, [r['extra_channels'][i][1] for r in region_results]))

    labels = [str(i + 1) for i in range(len(region_results))]
    colors = [REGION_COLORS[i % len(REGION_COLORS)] for i in range(len(region_results))]

    fig, axes = plt.subplots(1, len(channel_data), figsize=(4.0625, 3.75))
    axes = [axes] if len(channel_data) == 1 else axes
    for ax, (name, values_per_region) in zip(axes, channel_data):
        bp = ax.boxplot(values_per_region, labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Region', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, axis='y', linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def build_umap_figure(umap_df, group, task, model_name):
    fig = go.Figure(data=[
        go.Scatter(
            x=umap_df['umap1'],
            y=umap_df['umap2'],
            mode='markers',
            marker=dict(size=8, color=umap_df['class_numeric'], colorscale='Viridis'),
            customdata=umap_df[['age', 'filename', 'heatmap_path']].values,
            hovertemplate=(
                'Age: %{customdata[0]}<br>'
                'Coordinates: (%{x:.2f}, %{y:.2f})<br>'
                'Image Name: %{customdata[1]}<extra></extra>'
            ),
        )
    ])
    fig.update_layout(
        template='plotly_white',
        title=dict(text=f'Group: {group}  ·  Task: {task}  ·  Model: {model_name}', font=dict(size=15)),
        xaxis_title='UMAP1',
        yaxis_title='UMAP2',
        font=dict(family='-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        height=560,
    )
    return fig


# --- app / layout ------------------------------------------------------

def main_page_layout(default_base_dir):
    initial_groups = list_groups(default_base_dir) if is_valid_base_dir(default_base_dir) else []

    return html.Div(className='app-shell', children=[
        html.Div(className='app-header', children=[
            html.H2('NoRI Interactive Regression Explorer'),
            html.P('Explore UMAP embeddings, prediction quality, and attention scores across aging groups.'),
        ]),

        html.Div(className='card', children=[
            html.Div('Data source', className='card-title'),
            html.Label('Data folder (contains outputs/ and data/)', className='field-label'),
            html.Div(className='data-folder-row', children=[
                dcc.Input(
                    id='base-dir-input',
                    type='text',
                    value=default_base_dir,
                    debounce=True,
                    className='dash-input',
                ),
                html.Button('Load', id='base-dir-button', n_clicks=0, className='btn-primary'),
            ]),
            html.Div(id='base-dir-status', className='status-text'),
        ]),

        dcc.Store(id='base-dir-store', data=default_base_dir if is_valid_base_dir(default_base_dir) else None),

        html.Div(className='card', children=[
            html.Div('Selection', className='card-title'),
            html.Div(className='field-row', children=[
                html.Div([
                    html.Label('Group', className='field-label'),
                    dcc.Dropdown(
                        id='group-dropdown',
                        options=[{'label': g, 'value': g} for g in initial_groups],
                        value=initial_groups[0] if initial_groups else None,
                    ),
                ]),
                html.Div([
                    html.Label('Task', className='field-label'),
                    dcc.Dropdown(id='task-dropdown'),
                ]),
                html.Div([
                    html.Label('Model', className='field-label'),
                    dcc.Dropdown(id='model-dropdown'),
                ]),
            ]),
            html.Div(className='btn-secondary', style={'display': 'flex', 'gap': '10px'}, children=[
                html.Button('Show UMAP', id='show-umap-button', n_clicks=0, className='btn-primary'),
                html.A(
                    'Show all model plots', id='view-all-link', href='#', target='_blank',
                    className='btn-primary', style={'background': 'var(--color-text-muted)'},
                ),
                html.A(
                    'Whole-image overlay', id='overlay-link', href='#', target='_blank',
                    className='btn-primary', style={'background': 'var(--color-text-muted)'},
                ),
                html.A(
                    'Image viewer', id='image-viewer-link', href='#', target='_blank',
                    className='btn-primary', style={'background': 'var(--color-text-muted)'},
                ),
                html.A(
                    'Features', id='features-link', href='#', target='_blank',
                    className='btn-primary', style={'background': 'var(--color-text-muted)'},
                ),
                html.A(
                    'Statistics', id='statistics-link', href='#', target='_blank',
                    className='btn-primary', style={'background': 'var(--color-text-muted)'},
                ),
            ]),
        ]),

        html.Div(className='card', children=[
            html.Div('Model metrics', className='card-title'),
            html.Pre(id='metrics-output', className='metrics-box'),
        ]),

        html.Div(className='card', children=[
            html.Div('Attention scores by aging group', className='card-title'),
            plot_with_toolbar(html.Img(id='attn-boxplots-img', className='boxplot-img'), 'attention_boxplots'),
        ]),

        dcc.Store(id='umap-store'),

        html.Div(className='card', children=[
            html.Div('UMAP embedding', className='card-title'),
            dcc.Graph(id='umap-scatter'),
        ]),

        html.Div(className='card', children=[
            html.Div('Selected tile', className='card-title'),
            html.Div(
                id='image-panel-output',
                children=html.Div('Click a point in the UMAP plot to inspect a tile.', className='tile-placeholder'),
            ),
        ]),
    ])


TRAINING_PLOT_PREFIXES = [
    ('epoch_loss', 'Validation loss'),
    ('mae', 'MAE'),
    ('r2', 'R²'),
]


def find_model_plot_path(base_dir, group, task, prefix, model_name):
    models_dir = os.path.join(base_dir, 'outputs', group, task, 'models')
    if not os.path.isdir(models_dir):
        return None

    target_name = f'{prefix}_{model_name}'
    for fname in os.listdir(models_dir):
        if os.path.splitext(fname)[0] == target_name:
            return os.path.join(models_dir, fname)
    return None


def build_model_card(base_dir, group, task, model_name):
    children = [html.Div(model_name, className='umap-card-title')]

    try:
        umap_df = load_umap_df(base_dir, group, task, model_name)
        img_src = build_static_umap_image(umap_df, model_name)
        children.append(plot_with_toolbar(
            html.Img(src=img_src, className='umap-card-img'), f'{model_name}_umap',
        ))
        kde_src = build_static_umap_kde_image(umap_df, model_name)
        children.append(plot_with_toolbar(
            html.Img(src=kde_src, className='umap-card-img umap-card-kde'), f'{model_name}_umap_density',
        ))
        prediction_src = build_static_umap_prediction_image(umap_df, model_name)
        children.append(plot_with_toolbar(
            html.Img(src=prediction_src, className='umap-card-img umap-card-kde'), f'{model_name}_umap_prediction',
        ))
    except Exception:
        children.append(html.Pre(traceback.format_exc(), className='metrics-box'))

    training_imgs = []
    for prefix, label in TRAINING_PLOT_PREFIXES:
        path = find_model_plot_path(base_dir, group, task, prefix, model_name)
        if not path:
            continue
        try:
            data_uri = pil_to_data_uri(Image.open(path))
        except Exception:
            continue
        training_imgs.append(html.Div(className='training-img-wrap', children=[
            html.Div(label, className='training-img-label'),
            plot_with_toolbar(html.Img(src=data_uri, className='training-img'), f'{prefix}_{model_name}'),
        ]))

    if training_imgs:
        children.append(html.Div(training_imgs, className='training-img-row'))

    return html.Div(children, className='card')


def all_umaps_page_layout(base_dir, group, task):
    if not (base_dir and group and task and is_valid_base_dir(base_dir)):
        return html.Div(className='app-shell', children=[
            html.A('← Back to explorer', href='/', className='btn-outline'),
            html.Div(className='app-header', children=[
                html.H2('All model plots'),
                html.P('Missing or invalid data folder / group / task. Go back and select them first.'),
            ]),
        ])

    try:
        model_names = list_models(base_dir, group, task)
    except FileNotFoundError:
        model_names = []

    # Every model's card is built and present in the page, but only one is shown at a
    # time (client-side, see the "All model plots" section of plot_toolbar.js) — the
    # Prev/Next buttons below just toggle which .model-card is visible.
    cards = [
        html.Div(
            build_model_card(base_dir, group, task, model_name),
            className='model-card',
            style={'display': 'block'} if i == 0 else {'display': 'none'},
            **{'data-model-name': model_name},
        )
        for i, model_name in enumerate(model_names)
    ]

    nav_section = []
    if model_names:
        nav_section.append(html.Div(
            className='field-row',
            style={'alignItems': 'center', 'marginBottom': '16px'},
            children=[
                html.Button(
                    '←', id='all-models-prev-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Div(
                    f'{model_names[0]}  (1 / {len(model_names)})',
                    id='all-models-nav-label', className='status-text', style={'marginTop': 0},
                ),
                html.Button(
                    '→', id='all-models-next-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
            ],
        ))

    if not cards:
        cards = [html.Div("No models found in this task's umap/ folder.", className='tile-placeholder')]

    combined_section = []
    if model_names:
        try:
            combined_src = build_combined_prediction_boxplot(base_dir, group, task, model_names)
        except Exception:
            combined_src = None
            combined_error = traceback.format_exc()
        else:
            combined_error = None

        combined_section.append(html.Div(className='card', children=[
            html.Div('Predictions by age and sample (all models combined)', className='card-title'),
            plot_with_toolbar(
                html.Img(src=combined_src, className='umap-card-img umap-card-img-75'), 'predictions_by_age_and_sample',
            ) if combined_src
            else html.Pre(combined_error or 'No data available.', className='metrics-box'),
        ]))

        try:
            animal_level_src = build_animal_level_separation_plot(base_dir, group, task, model_names)
        except Exception:
            animal_level_src = None
            animal_level_error = traceback.format_exc()
        else:
            animal_level_error = None

        combined_section.append(html.Div(className='card', children=[
            html.Div('Animal-level distribution separation by age transition', className='card-title'),
            plot_with_toolbar(
                html.Img(src=animal_level_src, className='umap-card-img umap-card-img-75'), 'animal_level_separation',
            ) if animal_level_src
            else html.Pre(animal_level_error or 'No data available.', className='metrics-box'),
        ]))

        try:
            age_trend_src = build_animal_median_age_trend_plot(base_dir, group, task, model_names)
        except Exception:
            age_trend_src = None
            age_trend_error = traceback.format_exc()
        else:
            age_trend_error = None

        combined_section.append(html.Div(className='card', children=[
            html.Div('Median predicted age by animal (dots colored by model)', className='card-title'),
            plot_with_toolbar(
                html.Img(src=age_trend_src, className='umap-card-img umap-card-img-75'), 'median_predicted_age_by_animal',
            ) if age_trend_src
            else html.Pre(age_trend_error or 'No data available.', className='metrics-box'),
        ]))

    return html.Div(className='app-shell', children=[
        html.A('← Back to explorer', href='/', className='btn-outline'),
        html.Div(className='app-header', children=[
            html.H2(f'All model plots — {group} / {task}'),
            html.P(f'{len(model_names)} model(s) found.'),
        ]),
        *combined_section,
        *nav_section,
        html.Div(cards),
    ])


def features_page_layout(base_dir, group, task):
    if not (base_dir and group and task and is_valid_base_dir(base_dir)):
        return html.Div(className='app-shell', children=[
            html.A('← Back to explorer', href='/', className='btn-outline'),
            html.Div(className='app-header', children=[
                html.H2('Features'),
                html.P('Missing or invalid data folder / group / task. Go back and select them first.'),
            ]),
        ])

    feat_dir = features_dir_path(base_dir, group, task)
    available_kinds = list_feature_kinds_available(base_dir, group, task)
    default_kind = available_kinds[0] if available_kinds else 'main'

    try:
        columns = get_feature_columns(base_dir, group, task, default_kind)
    except Exception:
        columns = []
    column_options = [{'label': c, 'value': c} for c in columns]
    hue_options = [{'label': '(none)', 'value': ''}] + column_options
    default_x = columns[0] if columns else None
    default_y = columns[1] if len(columns) > 1 else default_x

    def column_picker_row(prefix):
        return html.Div(className='field-row', children=[
            html.Div([
                html.Label('X', className='field-label'),
                dcc.Dropdown(id=f'{prefix}-x-dropdown', options=column_options, value=default_x),
            ]),
            html.Div([
                html.Label('Y', className='field-label'),
                dcc.Dropdown(id=f'{prefix}-y-dropdown', options=column_options, value=default_y),
            ]),
            html.Div([
                html.Label('Hue', className='field-label'),
                dcc.Dropdown(id=f'{prefix}-hue-dropdown', options=hue_options, value=''),
            ]),
        ])

    pred_trend_features = list_pred_age_trend_features(base_dir, group, task)
    tubule_types = list_tubule_types(base_dir, group, task)
    tubule_type_options = [{'label': 'All tubules', 'value': TUBULE_TYPE_ALL}] + [
        {'label': t, 'value': t} for t in tubule_types
    ]
    default_tubule_type = 'proximal' if 'proximal' in tubule_types else TUBULE_TYPE_ALL

    return html.Div(className='app-shell', children=[
        html.A('← Back to explorer', href='/', className='btn-outline'),
        html.Div(className='app-header', children=[
            html.H2('Features'),
            html.P(f'{group} / {task}'),
        ]),

        dcc.Store(id='features-context-store', data={
            'base_dir': base_dir, 'group': group, 'task': task,
        }),

        html.Div(className='card', children=[
            html.Div('Selection', className='card-title'),
            html.Div(f'Looking for feature files in: {feat_dir}', className='status-text'),
            html.Div(className='field-row', style={'marginTop': '14px'}, children=[
                html.Div([
                    html.Label('Feature file', className='field-label'),
                    dcc.Dropdown(
                        id='features-kind-dropdown',
                        options=[{'label': label, 'value': kind} for kind, label in FEATURE_KINDS.items()],
                        value=default_kind,
                    ),
                ]),
                html.Div([
                    html.Label('Aggregation level', className='field-label'),
                    dcc.Dropdown(
                        id='features-agg-dropdown',
                        options=[
                            {'label': label, 'value': level}
                            for level, label in FEATURE_AGGREGATION_LEVELS.items()
                        ],
                        value='tubule',
                        clearable=False,
                    ),
                ]),
            ]),
            html.Div(id='features-status', className='status-text', style={'marginTop': '10px'}),
        ]),

        html.Div(className='card', children=[
            html.Div('Scatter plot', className='card-title'),
            column_picker_row('scatter'),
            html.Button(
                'Plot scatter', id='features-scatter-button', n_clicks=0,
                className='btn-primary', style={'marginTop': '10px'},
            ),
            html.Div(id='features-scatter-status', className='status-text', style={'marginTop': '10px'}),
            dcc.Graph(id='features-scatter-graph'),
        ]),

        html.Div(className='card', children=[
            html.Div('Selected point — protein / lipid crop', className='card-title'),
            html.Div(
                id='features-tile-panel',
                children=html.Div(
                    'Click a point in the scatter plot to inspect its protein/lipid crop.',
                    className='tile-placeholder',
                ),
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Boxplot / violin plot', className='card-title'),
            html.Div(className='field-row', children=[
                html.Div([
                    html.Label('Plot type', className='field-label'),
                    dcc.Dropdown(
                        id='box-plot-type-dropdown',
                        options=[
                            {'label': 'Boxplot', 'value': 'box'},
                            {'label': 'Violin plot', 'value': 'violin'},
                        ],
                        value='box',
                        clearable=False,
                    ),
                ]),
            ]),
            column_picker_row('box'),
            html.Button(
                'Plot', id='features-box-button', n_clicks=0,
                className='btn-primary', style={'marginTop': '10px'},
            ),
            html.Div(id='features-box-status', className='status-text', style={'marginTop': '10px'}),
            plot_with_toolbar(
                html.Img(id='features-box-img', className='umap-card-img umap-card-img-75'),
                'features_boxplot',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Predicted-age trend', className='card-title'),
            html.Div(
                'For tubules with a valid pred_320 in [5, 28): trims outliers (IQR rule) on both pred_320 '
                'and the feature, bins pred_320 into 0.2-wide buckets and averages within each bucket, then '
                "plots the trend line and its Spearman correlation. Combines the main tubule table with the "
                'shape table (joined on contour_id / label_id), so both kinds of feature are selectable here.',
                className='status-text', style={'marginBottom': '10px'},
            ),
            html.Div(className='field-row', children=[
                html.Div([
                    html.Label('Feature', className='field-label'),
                    dcc.Dropdown(
                        id='pred-trend-feature-dropdown',
                        options=[{'label': f, 'value': f} for f in pred_trend_features],
                        value=pred_trend_features[0] if pred_trend_features else None,
                    ),
                ]),
                html.Div([
                    html.Label('Tubule type', className='field-label'),
                    dcc.Dropdown(
                        id='pred-trend-tubule-dropdown',
                        options=tubule_type_options,
                        value=default_tubule_type,
                        clearable=False,
                    ),
                ]),
            ]),
            html.Button(
                'Show plot', id='pred-trend-button', n_clicks=0,
                className='btn-primary', style={'marginTop': '10px'},
            ),
            html.Div(id='pred-trend-status', className='status-text', style={'marginTop': '10px'}),
            plot_with_toolbar(
                html.Img(id='pred-trend-img', className='umap-card-img umap-card-img-75'), 'pred_age_trend',
            ),
        ]),
    ])


def statistics_page_layout(base_dir, group, task):
    if not (base_dir and group and task and is_valid_base_dir(base_dir)):
        return html.Div(className='app-shell', children=[
            html.A('← Back to explorer', href='/', className='btn-outline'),
            html.Div(className='app-header', children=[
                html.H2('Statistics'),
                html.P('Missing or invalid data folder / group / task. Go back and select them first.'),
            ]),
        ])

    available_kinds = list_feature_kinds_available(base_dir, group, task)
    default_kind = available_kinds[0] if available_kinds else 'main'

    return html.Div(className='app-shell', children=[
        html.A('← Back to explorer', href='/', className='btn-outline'),
        html.Div(className='app-header', children=[
            html.H2('Statistics'),
            html.P(f'{group} / {task}'),
        ]),

        dcc.Store(id='stats-context-store', data={
            'base_dir': base_dir, 'group': group, 'task': task,
        }),
        dcc.Store(id='stats-results-store'),

        html.Div(className='card', children=[
            html.Div('Selection', className='card-title'),
            html.Div(className='field-row', children=[
                html.Div([
                    html.Label('Feature file', className='field-label'),
                    dcc.Dropdown(
                        id='stats-kind-dropdown',
                        options=[{'label': label, 'value': kind} for kind, label in FEATURE_KINDS.items()],
                        value=default_kind,
                    ),
                ]),
                html.Div([
                    html.Label('Aggregation level', className='field-label'),
                    dcc.Dropdown(
                        id='stats-agg-dropdown',
                        options=[
                            {'label': label, 'value': level}
                            for level, label in FEATURE_AGGREGATION_LEVELS.items()
                        ],
                        value='animal_median',
                        clearable=False,
                    ),
                ]),
            ]),
            html.Div(
                f'Runs every test below for every numeric feature: Kruskal-Wallis and Welch\'s ANOVA (does '
                f'the feature differ across all age groups at once, parametrically and non-parametrically), '
                f'Spearman\'s ρ and a linear regression (is there a monotonic/linear trend with age), and a '
                f'random-intercept mixed-effects model (feature ~ age, animal as random effect — uses every '
                f'tubule while still accounting for which animal it came from). For each pair of adjacent '
                f'age groups: Welch\'s t-test and Mann-Whitney U (pairwise significance), Cohen\'s d and '
                f"Cliff's delta (matching effect sizes). Each test's p-values get their own "
                f'Benjamini-Hochberg FDR q-value across every feature tested. Animal-level aggregation is '
                f'recommended over raw tubules to avoid pseudoreplication (thousands of correlated tubules '
                f'from a handful of animals inflating significance)'
                + ('.' if STATSMODELS_AVAILABLE else ' — statsmodels is not installed, so the mixed-effects '
                   'model will be skipped.'),
                className='status-text', style={'marginTop': '10px'},
            ),
            html.Button(
                'Run tests', id='stats-run-button', n_clicks=0,
                className='btn-primary', style={'marginTop': '10px'},
            ),
            html.Div(id='stats-status', className='status-text', style={'marginTop': '10px'}),
        ]),

        html.Div(className='app-header', children=[
            html.H3('Differences across all ages'),
            html.P('Kruskal-Wallis, Welch\'s ANOVA, Spearman\'s ρ, linear regression, mixed-effects model.'),
        ]),

        html.Div(className='card', children=[
            html.Div('Ranked features (overall age effect)', className='card-title'),
            plot_with_toolbar(
                html.Img(id='stats-overall-ranked-img', className='umap-card-img umap-card-img-75'),
                'stats_overall_ranked_features',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Effect size vs. significance (volcano plot)', className='card-title'),
            plot_with_toolbar(
                html.Img(id='stats-overall-volcano-img', className='umap-card-img umap-card-img-75'),
                'stats_overall_volcano',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Results table', className='card-title'),
            html.Div(id='stats-overall-table-container'),
        ]),

        html.Div(className='app-header', children=[
            html.H3('Differences between age groups'),
            html.P("Welch's t-test, Mann-Whitney U, Cohen's d, Cliff's delta — one row per adjacent age transition."),
        ]),

        html.Div(className='card', children=[
            html.Div('Age-transition effect-size heatmap', className='card-title'),
            html.Div(
                "Rows = features, columns = age transitions, color/value = Cohen's d — see at a glance which "
                'features move most, in which direction, and whether the change is concentrated in one '
                'transition or spread across the whole age range.',
                className='status-text', style={'marginBottom': '10px'},
            ),
            plot_with_toolbar(
                html.Img(id='stats-pairwise-heatmap-img', className='umap-card-img umap-card-img-50'),
                'stats_pairwise_heatmap',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Ranked features (pairwise age transitions)', className='card-title'),
            plot_with_toolbar(
                html.Img(id='stats-pairwise-ranked-img', className='umap-card-img umap-card-img-75'),
                'stats_pairwise_ranked_features',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Effect size vs. significance (volcano plot)', className='card-title'),
            plot_with_toolbar(
                html.Img(id='stats-pairwise-volcano-img', className='umap-card-img umap-card-img-75'),
                'stats_pairwise_volcano',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Results table', className='card-title'),
            html.Div(id='stats-pairwise-table-container'),
        ]),

        html.Div(className='card', children=[
            html.Div('Inspect a feature', className='card-title'),
            html.Div(className='field-row', children=[
                html.Div([
                    html.Label('Feature', className='field-label'),
                    dcc.Dropdown(id='stats-inspect-dropdown'),
                ]),
                html.Div([
                    html.Label('Plot type', className='field-label'),
                    dcc.Dropdown(
                        id='stats-inspect-type-dropdown',
                        options=[
                            {'label': 'Boxplot', 'value': 'box'},
                            {'label': 'Violin plot', 'value': 'violin'},
                        ],
                        value='box', clearable=False,
                    ),
                ]),
            ]),
            html.Button(
                'Show plot', id='stats-inspect-button', n_clicks=0,
                className='btn-primary', style={'marginTop': '10px'},
            ),
            html.Div(id='stats-inspect-status', className='status-text', style={'marginTop': '10px'}),
            plot_with_toolbar(
                html.Img(id='stats-inspect-img', className='umap-card-img umap-card-img-75'), 'stats_feature_inspect',
            ),
        ]),
    ])


def whole_image_overlay_page_layout(base_dir, group, task, model):
    if not (base_dir and group and task and model and is_valid_base_dir(base_dir)):
        return html.Div(className='app-shell', children=[
            html.A('← Back to explorer', href='/', className='btn-outline'),
            html.Div(className='app-header', children=[
                html.H2('Whole-image prediction overlay'),
                html.P('Missing or invalid data folder / group / task / model. Go back and select them first.'),
            ]),
        ])

    try:
        whole_images = list_whole_images(base_dir, group, task, model)
        list_error = None
    except Exception:
        whole_images = []
        list_error = traceback.format_exc()

    data_group_dir = os.path.join(base_dir, 'data', group)

    return html.Div(className='app-shell', children=[
        html.A('← Back to explorer', href='/', className='btn-outline'),
        html.Div(className='app-header', children=[
            html.H2('Whole-image prediction overlay'),
            html.P(f'{group} / {task} / {model}'),
        ]),

        dcc.Store(id='overlay-context-store', data={
            'base_dir': base_dir, 'group': group, 'task': task, 'model': model,
        }),

        html.Div(className='card', children=[
            html.Div('Selection', className='card-title'),
            html.Div(f'Looking for images in: {data_group_dir}', className='status-text'),
            html.Div(className='field-row', style={'marginTop': '14px'}, children=[
                html.Div([
                    html.Label('Image', className='field-label'),
                    dcc.Dropdown(
                        id='overlay-image-dropdown',
                        options=[{'label': img, 'value': img} for img in whole_images],
                        value=whole_images[0] if whole_images else None,
                    ),
                ]),
            ]),
            html.Pre(list_error, className='metrics-box') if list_error else None,
            html.Button(
                'Show overlay', id='overlay-show-button', n_clicks=0,
                className='btn-primary btn-secondary',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Overlay', className='card-title'),
            html.Div(id='overlay-status', className='status-text'),
            html.Div([
                'Left-click the image to zoom in, right-click to zoom out (10% per click). ',
                html.Span('Zoom: 100%', id='viewer-zoom-readout'),
            ], className='status-text'),
            html.Div(className='field-row', style={'marginTop': '10px', 'alignItems': 'center'}, children=[
                html.Button(
                    'Inspect tile', id='overlay-inspect-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Div(
                    'When on, clicking the image (instead of zooming) looks up that tile\'s '
                    'heatmap(s) below, one per model that has one.',
                    className='status-text', style={'marginTop': 0},
                ),
            ]),
            dcc.Input(id='overlay-tile-click-input', type='text', value='', style={'display': 'none'}),
            html.Div(className='viewer-image-row', style={'marginTop': '12px'}, children=[
                html.Div(className='plot-wrap', children=[
                    html.Div(className='plot-toolbar', children=[
                        html.Button(
                            '⎘', title='Copy image', className='plot-icon-btn plot-copy-btn',
                            **{'data-filename': 'whole_image_overlay'},
                        ),
                        html.Button(
                            '⬇', title='Download image', className='plot-icon-btn plot-download-btn',
                            **{'data-filename': 'whole_image_overlay'},
                        ),
                    ]),
                    html.Div(className='zoom-scroll', children=[
                        html.Div(className='zoom-image-wrap', children=[
                            html.Img(
                                id='overlay-image', className='zoom-image',
                                **{'data-image-path': ''},
                            ),
                        ]),
                    ]),
                ]),
                html.Img(id='overlay-colorbar', className='viewer-colorbars-img'),
            ]),
        ]),

        html.Div(className='card', children=[
            html.Div('Selected tile — heatmaps', className='card-title'),
            html.Div(
                'Turn on "Inspect tile" above and click a point on the overlay image to '
                'load that tile\'s heatmap(s) here.',
                id='tile-heatmap-status', className='status-text',
            ),
            dcc.Store(id='tile-heatmap-store', data={'items': [], 'index': 0}),
            html.Div(id='tile-heatmap-carousel', style={'display': 'none'}, children=[
                html.Div(className='field-row', style={'marginTop': '10px', 'alignItems': 'center'}, children=[
                    html.Button(
                        '←', id='tile-heatmap-prev-button', n_clicks=0,
                        className='btn-outline', style={'flex': '0 0 auto'},
                    ),
                    html.Div(id='tile-heatmap-label', className='status-text', style={'marginTop': 0}),
                    html.Button(
                        '→', id='tile-heatmap-next-button', n_clicks=0,
                        className='btn-outline', style={'flex': '0 0 auto'},
                    ),
                ]),
                plot_with_toolbar(
                    html.Img(id='tile-heatmap-img', className='tile-image', style={'marginTop': '10px'}),
                    'tile_heatmap',
                ),
            ]),
        ]),
    ])


def image_viewer_page_layout(base_dir, group, task):
    if not (base_dir and group and task and is_valid_base_dir(base_dir)):
        return html.Div(className='app-shell', children=[
            html.A('← Back to explorer', href='/', className='btn-outline'),
            html.Div(className='app-header', children=[
                html.H2('Image viewer'),
                html.P('Missing or invalid data folder / group / task. Go back and select them first.'),
            ]),
        ])

    images = list_data_tif_images(base_dir, group)
    data_group_dir = os.path.join(base_dir, 'data', group)
    channels_path = channels_txt_path(base_dir, group, task)
    channels_found = os.path.isfile(channels_path)
    channel_names = load_channel_names(base_dir, group, task)
    channel_options = [{'label': f'{idx}: {name}', 'value': idx} for idx, name in sorted(channel_names.items())]
    default_channel_idxs = sorted({
        find_channel_index(channel_names, 'protein', 0),
        find_channel_index(channel_names, 'lipid', 1),
    })

    if channels_found:
        channels_status = f'channels.txt: {channels_path}  ·  {len(channel_names)} channel(s) loaded'
        channels_status_class = 'status-text'
    else:
        channels_status = f'channels.txt not found at: {channels_path}  ·  showing protein/lipid only'
        channels_status_class = 'status-text status-error'

    return html.Div(className='app-shell', children=[
        html.A('← Back to explorer', href='/', className='btn-outline'),
        html.Div(className='app-header', children=[
            html.H2('Image viewer'),
            html.P(f'{group} / {task}'),
        ]),

        dcc.Store(id='viewer-context-store', data={
            'base_dir': base_dir, 'group': group, 'task': task,
        }),

        html.Div(className='card', children=[
            html.Div('Selection', className='card-title'),
            html.Div(f'Looking for images in: {data_group_dir}', className='status-text'),
            html.Div(channels_status, className=channels_status_class),
            html.Div(className='field-row', style={'marginTop': '14px'}, children=[
                html.Div([
                    html.Label('Image', className='field-label'),
                    dcc.Dropdown(
                        id='viewer-image-dropdown',
                        options=[{'label': img, 'value': img} for img in images],
                        value=images[0] if images else None,
                    ),
                ]),
                html.Div([
                    html.Label('Channels (1 or more)', className='field-label'),
                    dcc.Dropdown(
                        id='viewer-channel-dropdown', options=channel_options,
                        value=default_channel_idxs, multi=True,
                    ),
                ]),
            ]),
            html.Div([
                html.Label('Channel normalization', className='field-label'),
                dcc.RadioItems(
                    id='viewer-norm-mode',
                    options=NORM_MODE_OPTIONS,
                    value=NORM_MODE_GLOBAL,
                    labelStyle={'display': 'block'},
                ),
            ], style={'marginTop': '14px'}),
            html.Div(
                'No .tif images found under this group.', className='status-text status-error',
            ) if not images else None,
            html.Button(
                'Show image', id='viewer-show-button', n_clicks=0,
                className='btn-primary btn-secondary',
            ),
        ]),

        html.Div(className='card', children=[
            html.Div('Image', className='card-title'),
            html.Div(id='viewer-status', className='status-text'),
            html.Div([
                'Left-click the image to zoom in, right-click to zoom out (10% per click). ',
                html.Span('Zoom: 100%', id='viewer-zoom-readout'),
            ], className='status-text'),
            html.Div(className='field-row', style={'marginTop': '10px', 'alignItems': 'center'}, children=[
                html.Button(
                    'Rectangle', id='viewer-select-rect-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Button(
                    'Polygon', id='viewer-select-polygon-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Button(
                    'Clear regions', id='viewer-clear-regions-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Div(
                    'Rectangle: drag to draw. Polygon: click each vertex, then double-click '
                    "(or click the first vertex) to close it, Esc to cancel. Repeat to pick "
                    "several regions; each one's protein/lipid distribution is plotted below.",
                    className='status-text', style={'marginTop': 0},
                ),
            ]),
            dcc.Input(id='viewer-region-input', type='text', value='[]', style={'display': 'none'}),
            html.Div(className='viewer-image-row', style={'marginTop': '12px'}, children=[
                html.Div(className='plot-wrap', children=[
                    html.Div(className='plot-toolbar', children=[
                        html.Button(
                            '⎘', title='Copy image', className='plot-icon-btn plot-copy-btn',
                            **{'data-filename': 'nori_image'},
                        ),
                        html.Button(
                            '⬇', title='Download image', className='plot-icon-btn plot-download-btn',
                            **{'data-filename': 'nori_image'},
                        ),
                    ]),
                    html.Div(className='zoom-scroll', children=[
                        html.Div(className='zoom-image-wrap', children=[
                            html.Img(
                                id='viewer-image', className='zoom-image',
                                **{'data-image-path': ''},
                            ),
                        ]),
                    ]),
                ]),
                html.Img(id='viewer-channel-colorbars', className='viewer-colorbars-img'),
            ]),
        ]),

        html.Div(className='card', children=[
            html.Div('Selected regions — protein / lipid distribution', className='card-title'),
            html.Div(id='region-plot-status', className='status-text'),
            html.Div(id='viewer-regions-container', className='region-list'),
        ]),
    ])


def create_app(default_base_dir=None):
    assets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    app = dash.Dash(__name__, assets_folder=assets_folder, suppress_callback_exceptions=True)
    app.title = 'NoRI Regression Explorer'

    default_base_dir = default_base_dir or os.getcwd()

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content'),
    ])

    register_callbacks(app, default_base_dir)
    return app


def register_callbacks(app, default_base_dir):

    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname'),
        Input('url', 'search'),
    )
    def render_page(pathname, search):
        params = parse_qs((search or '').lstrip('?'))
        base_dir = params.get('base_dir', [''])[0]
        group = params.get('group', [''])[0]
        task = params.get('task', [''])[0]

        if pathname == '/all-umaps':
            return all_umaps_page_layout(base_dir, group, task)
        if pathname == '/whole-image-overlay':
            model = params.get('model', [''])[0]
            return whole_image_overlay_page_layout(base_dir, group, task, model)
        if pathname == '/image-viewer':
            return image_viewer_page_layout(base_dir, group, task)
        if pathname == '/features':
            return features_page_layout(base_dir, group, task)
        if pathname == '/statistics':
            return statistics_page_layout(base_dir, group, task)
        return main_page_layout(default_base_dir)

    @app.callback(
        Output('view-all-link', 'href'),
        Input('base-dir-store', 'data'),
        Input('group-dropdown', 'value'),
        Input('task-dropdown', 'value'),
    )
    def update_view_all_link(base_dir, group, task):
        if not (base_dir and group and task):
            return '#'
        query = urlencode({'base_dir': base_dir, 'group': group, 'task': task})
        return f'/all-umaps?{query}'

    @app.callback(
        Output('overlay-link', 'href'),
        Input('base-dir-store', 'data'),
        Input('group-dropdown', 'value'),
        Input('task-dropdown', 'value'),
        Input('model-dropdown', 'value'),
    )
    def update_overlay_link(base_dir, group, task, model):
        if not (base_dir and group and task and model):
            return '#'
        query = urlencode({'base_dir': base_dir, 'group': group, 'task': task, 'model': model})
        return f'/whole-image-overlay?{query}'

    @app.callback(
        Output('image-viewer-link', 'href'),
        Input('base-dir-store', 'data'),
        Input('group-dropdown', 'value'),
        Input('task-dropdown', 'value'),
    )
    def update_image_viewer_link(base_dir, group, task):
        if not (base_dir and group and task):
            return '#'
        query = urlencode({'base_dir': base_dir, 'group': group, 'task': task})
        return f'/image-viewer?{query}'

    @app.callback(
        Output('features-link', 'href'),
        Input('base-dir-store', 'data'),
        Input('group-dropdown', 'value'),
        Input('task-dropdown', 'value'),
    )
    def update_features_link(base_dir, group, task):
        if not (base_dir and group and task):
            return '#'
        query = urlencode({'base_dir': base_dir, 'group': group, 'task': task})
        return f'/features?{query}'

    @app.callback(
        Output('statistics-link', 'href'),
        Input('base-dir-store', 'data'),
        Input('group-dropdown', 'value'),
        Input('task-dropdown', 'value'),
    )
    def update_statistics_link(base_dir, group, task):
        if not (base_dir and group and task):
            return '#'
        query = urlencode({'base_dir': base_dir, 'group': group, 'task': task})
        return f'/statistics?{query}'

    @app.callback(
        Output('features-status', 'children'),
        Output('scatter-x-dropdown', 'options'),
        Output('scatter-y-dropdown', 'options'),
        Output('scatter-hue-dropdown', 'options'),
        Output('box-x-dropdown', 'options'),
        Output('box-y-dropdown', 'options'),
        Output('box-hue-dropdown', 'options'),
        Output('scatter-x-dropdown', 'value'),
        Output('scatter-y-dropdown', 'value'),
        Output('scatter-hue-dropdown', 'value'),
        Output('box-x-dropdown', 'value'),
        Output('box-y-dropdown', 'value'),
        Output('box-hue-dropdown', 'value'),
        Input('features-kind-dropdown', 'value'),
        Input('features-agg-dropdown', 'value'),
        State('features-context-store', 'data'),
    )
    def update_feature_columns(kind, level, context):
        no_updates = (dash.no_update,) * 12
        if not (kind and context):
            return (html.Span('Select a feature file.', className='status-text'), *no_updates)

        try:
            columns = get_feature_columns(
                context['base_dir'], context['group'], context['task'], kind, level or 'tubule',
            )
            n_files = len(list_feature_files(context['base_dir'], context['group'], context['task'], kind))
        except Exception:
            return (html.Pre(traceback.format_exc(), className='metrics-box'), *no_updates)

        if not columns:
            empty_hue = [{'label': '(none)', 'value': ''}]
            status = html.Span(
                f'No feature files found for this selection in {features_dir_path(context["base_dir"], context["group"], context["task"])}',
                className='status-error',
            )
            return (
                status, [], [], empty_hue, [], [], empty_hue,
                None, None, '', None, None, '',
            )

        column_options = [{'label': c, 'value': c} for c in columns]
        hue_options = [{'label': '(none)', 'value': ''}] + column_options
        default_x = columns[0]
        default_y = columns[1] if len(columns) > 1 else columns[0]
        status = html.Span(f'{len(columns)} column(s) across {n_files} image(s).', className='status-ok')

        return (
            status,
            column_options, column_options, hue_options,
            column_options, column_options, hue_options,
            default_x, default_y, '',
            default_x, default_y, '',
        )

    @app.callback(
        Output('features-scatter-status', 'children'),
        Output('features-scatter-graph', 'figure'),
        Input('features-scatter-button', 'n_clicks'),
        State('features-kind-dropdown', 'value'),
        State('features-agg-dropdown', 'value'),
        State('scatter-x-dropdown', 'value'),
        State('scatter-y-dropdown', 'value'),
        State('scatter-hue-dropdown', 'value'),
        State('features-context-store', 'data'),
        prevent_initial_call=True,
    )
    def update_features_scatter(n_clicks, kind, level, x_col, y_col, hue_col, context):
        if not (n_clicks and context and kind and x_col and y_col):
            return dash.no_update, dash.no_update

        try:
            df = load_feature_table(context['base_dir'], context['group'], context['task'], kind)
            df = aggregate_feature_table(df, level or 'tubule')
            if df.empty:
                return html.Span('No feature data found.', className='status-error'), go.Figure()
            fig = build_feature_scatter_figure(df, x_col, y_col, hue_col or None)
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), go.Figure()

        if fig is None:
            return html.Span('No rows with values for the selected columns.', className='status-error'), go.Figure()
        return html.Span(f'{len(df)} row(s) loaded.', className='status-ok'), fig

    @app.callback(
        Output('features-tile-panel', 'children'),
        Input('features-scatter-graph', 'clickData'),
        State('features-kind-dropdown', 'value'),
        State('scatter-x-dropdown', 'value'),
        State('scatter-y-dropdown', 'value'),
        State('features-context-store', 'data'),
    )
    def display_feature_tile(click_data, kind, x_col, y_col, context):
        placeholder = html.Div(
            'Click a point in the scatter plot to inspect its protein/lipid crop.',
            className='tile-placeholder',
        )
        if not click_data or not context:
            return placeholder

        point = click_data['points'][0]
        image_name, id_value, min_x, min_y, max_x, max_y = point['customdata']
        dot_x, dot_y = round_if_float(point['x']), round_if_float(point['y'])

        dot_coords_line = html.P([
            html.Strong('Image: '), image_name, '   ',
            html.Strong(f'{x_col}: '), f'{dot_x}   ',
            html.Strong(f'{y_col}: '), f'{dot_y}',
        ], className='tile-meta')

        bbox_missing = any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (min_x, min_y, max_x, max_y))
        if bbox_missing:
            if id_value is None or (isinstance(id_value, float) and np.isnan(id_value)):
                return html.Div([
                    dot_coords_line,
                    html.Div('No bounding-box coordinates available for this point.', className='status-error'),
                ])
            try:
                lookup = get_feature_bbox_lookup(context['base_dir'], context['group'], context['task'])
            except Exception:
                return html.Div([dot_coords_line, html.Pre(traceback.format_exc(), className='metrics-box')])

            bbox = lookup.get((image_name, id_value))
            if bbox is None:
                return html.Div([
                    dot_coords_line,
                    html.Div('No bounding-box coordinates available for this point.', className='status-error'),
                ])
            min_x, min_y, max_x, max_y = bbox

        try:
            crop_img, tif_path, (min_x, min_y, max_x, max_y) = build_feature_tile_crop_image(
                context['base_dir'], context['group'], context['task'], image_name, min_x, min_y, max_x, max_y,
            )
        except Exception:
            return html.Div([dot_coords_line, html.Pre(traceback.format_exc(), className='metrics-box')])

        return html.Div([
            dot_coords_line,
            html.P([
                html.Strong('Region: '), f'y[{min_y}:{max_y}], x[{min_x}:{max_x}]',
            ], className='tile-meta'),
            html.P(f'Source: {tif_path}', className='tile-meta'),
            plot_with_toolbar(
                html.Img(src=pil_to_data_uri(crop_img), className='tile-image'),
                f'{image_name}_crop_y{min_y}-{max_y}_x{min_x}-{max_x}',
            ),
        ])

    @app.callback(
        Output('features-box-status', 'children'),
        Output('features-box-img', 'src'),
        Input('features-box-button', 'n_clicks'),
        State('features-kind-dropdown', 'value'),
        State('features-agg-dropdown', 'value'),
        State('box-plot-type-dropdown', 'value'),
        State('box-x-dropdown', 'value'),
        State('box-y-dropdown', 'value'),
        State('box-hue-dropdown', 'value'),
        State('features-context-store', 'data'),
        prevent_initial_call=True,
    )
    def update_features_box(n_clicks, kind, level, plot_type, x_col, y_col, hue_col, context):
        if not (n_clicks and context and kind and x_col and y_col):
            return dash.no_update, dash.no_update

        try:
            df = load_feature_table(context['base_dir'], context['group'], context['task'], kind)
            df = aggregate_feature_table(df, level or 'tubule')
            if df.empty:
                return html.Span('No feature data found.', className='status-error'), None
            img_src = build_feature_box_or_violin_plot(df, x_col, y_col, hue_col or None, plot_type or 'box')
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), None

        if img_src is None:
            return html.Span('No rows with values for the selected columns.', className='status-error'), None
        return html.Span(f'{len(df)} row(s) loaded.', className='status-ok'), img_src

    @app.callback(
        Output('pred-trend-status', 'children'),
        Output('pred-trend-img', 'src'),
        Input('pred-trend-button', 'n_clicks'),
        State('pred-trend-feature-dropdown', 'value'),
        State('pred-trend-tubule-dropdown', 'value'),
        State('features-context-store', 'data'),
        prevent_initial_call=True,
    )
    def show_pred_age_trend(n_clicks, feature, tubule_type, context):
        if not (n_clicks and feature and context):
            return dash.no_update, dash.no_update

        try:
            df = load_main_shape_merged_table(context['base_dir'], context['group'], context['task'])
            if df.empty:
                return html.Span('No feature data found.', className='status-error'), None
            img_src, rho, p_value = build_pred_age_trend_plot(df, feature, tubule_type=tubule_type or TUBULE_TYPE_ALL)
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), None

        if img_src is None:
            tubule_note = 'any tubule type' if not tubule_type or tubule_type == TUBULE_TYPE_ALL else f'{tubule_type} tubules'
            return html.Span(
                f'Not enough data for this feature after filtering ({tubule_note}, pred_320 in '
                '[5, 28), outliers trimmed).', className='status-error',
            ), None

        sig_note = 'significant' if p_value <= 0.05 else 'not significant'
        status_class = 'status-ok' if p_value <= 0.05 else 'status-text'
        return html.Span(
            f'Spearman R={rho:.3f}, p={p_value:.3g} ({sig_note} at α = 0.05).', className=status_class,
        ), img_src

    @app.callback(
        Output('stats-status', 'children'),
        Output('stats-overall-ranked-img', 'src'),
        Output('stats-overall-volcano-img', 'src'),
        Output('stats-overall-table-container', 'children'),
        Output('stats-pairwise-heatmap-img', 'src'),
        Output('stats-pairwise-ranked-img', 'src'),
        Output('stats-pairwise-volcano-img', 'src'),
        Output('stats-pairwise-table-container', 'children'),
        Output('stats-inspect-dropdown', 'options'),
        Output('stats-inspect-dropdown', 'value'),
        Output('stats-results-store', 'data'),
        Input('stats-run-button', 'n_clicks'),
        State('stats-kind-dropdown', 'value'),
        State('stats-agg-dropdown', 'value'),
        State('stats-context-store', 'data'),
        prevent_initial_call=True,
    )
    def run_feature_stats(n_clicks, kind, level, context):
        no_updates = (dash.no_update,) * 10
        if not (n_clicks and context and kind):
            return (dash.no_update, *no_updates)

        try:
            df = load_feature_table(context['base_dir'], context['group'], context['task'], kind)
            df = aggregate_feature_table(df, level or 'animal_median')
            if df.empty or 'class_name' not in df.columns:
                status = html.Span(
                    'No feature data (or no class_name/age column) found for this selection.',
                    className='status-error',
                )
                return (status, *no_updates)
            overall = compute_overall_feature_stats(df, age_col='class_name')
            pairwise = compute_pairwise_feature_stats(df, age_col='class_name')
        except Exception:
            return (html.Pre(traceback.format_exc(), className='metrics-box'), *no_updates)

        if overall.empty and pairwise.empty:
            status = html.Span(
                'No numeric features could be tested (need at least 2 age groups with data).',
                className='status-error',
            )
            return (status, *no_updates)

        overall_ranked_src = build_stats_ranked_bar_plot(
            overall, title_suffix='features by overall age effect',
        )
        overall_volcano_src = build_stats_volcano_plot(
            overall, effect_col='abs_spearman_rho', transition_col=None,
            xlabel="Effect size (|Spearman ρ|)", title='Overall age effect: significance vs. trend strength',
        )
        overall_table = build_overall_stats_table(overall)

        pairwise_heatmap_src = build_pairwise_effect_size_heatmap(pairwise)
        pairwise_ranked_src = build_stats_ranked_bar_plot(
            pairwise, title_suffix='features by pairwise age-transition difference',
        )
        pairwise_volcano_src = build_stats_volcano_plot(pairwise)
        pairwise_table = build_pairwise_stats_table(pairwise)

        unique_features = list(dict.fromkeys([*overall['feature'], *pairwise['feature']]))
        feature_options = [{'label': f, 'value': f} for f in unique_features]
        n_sig_overall = int((overall['q_value'] < 0.05).sum()) if not overall.empty else 0
        n_sig_pairwise = int((pairwise['q_value'] < 0.05).sum()) if not pairwise.empty else 0
        status = html.Span(
            f'{len(overall)} feature(s) tested overall ({n_sig_overall} significant at q < 0.05) · '
            f'{len(pairwise)} feature-transition row(s) tested pairwise ({n_sig_pairwise} significant).',
            className='status-ok',
        )

        return (
            status,
            overall_ranked_src, overall_volcano_src, overall_table,
            pairwise_heatmap_src, pairwise_ranked_src, pairwise_volcano_src, pairwise_table,
            feature_options, unique_features[0] if unique_features else None,
            {'kind': kind, 'level': level or 'animal_median'},
        )

    @app.callback(
        Output('stats-inspect-status', 'children'),
        Output('stats-inspect-img', 'src'),
        Input('stats-inspect-button', 'n_clicks'),
        State('stats-inspect-dropdown', 'value'),
        State('stats-inspect-type-dropdown', 'value'),
        State('stats-results-store', 'data'),
        State('stats-context-store', 'data'),
        prevent_initial_call=True,
    )
    def show_stats_inspect_plot(n_clicks, feature, plot_type, results_meta, context):
        if not (n_clicks and feature and results_meta and context):
            return dash.no_update, dash.no_update

        try:
            df = load_feature_table(context['base_dir'], context['group'], context['task'], results_meta['kind'])
            df = aggregate_feature_table(df, results_meta.get('level', 'animal_median'))
            img_src = build_feature_box_or_violin_plot(df, 'class_name', feature, None, plot_type or 'box')
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), None

        if img_src is None:
            return html.Span('No rows with values for this feature.', className='status-error'), None
        return html.Span(f'Showing {feature} by class_name.', className='status-ok'), img_src

    TILE_HEATMAP_IDLE_STATUS = (
        'Turn on "Inspect tile" above and click a point on the overlay image to load '
        'that tile\'s heatmap(s) here.'
    )

    @app.callback(
        Output('overlay-status', 'children'),
        Output('overlay-image', 'src'),
        Output('overlay-colorbar', 'src'),
        Output('tile-heatmap-store', 'data', allow_duplicate=True),
        Output('tile-heatmap-status', 'children', allow_duplicate=True),
        Input('overlay-show-button', 'n_clicks'),
        State('overlay-image-dropdown', 'value'),
        State('overlay-context-store', 'data'),
        prevent_initial_call=True,
    )
    def update_overlay(n_clicks, image_name, context):
        if not (n_clicks and context and image_name):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        cleared_heatmaps = {'items': [], 'index': 0}

        try:
            img_src, tif_path, tile_size = build_whole_image_overlay(
                context['base_dir'], context['group'], context['task'], context['model'], image_name,
            )
        except Exception:
            return (
                html.Pre(traceback.format_exc(), className='metrics-box'), None, None,
                cleared_heatmaps, TILE_HEATMAP_IDLE_STATUS,
            )

        status = html.Span(
            f'Loaded: {tif_path}  ·  Tile size: {tile_size}px (auto-detected)', className='status-ok',
        )
        colorbar_img = build_overlay_colorbar()
        return (
            status, img_src, pil_to_data_uri(colorbar_img),
            cleared_heatmaps, TILE_HEATMAP_IDLE_STATUS,
        )

    @app.callback(
        Output('tile-heatmap-store', 'data', allow_duplicate=True),
        Output('tile-heatmap-status', 'children', allow_duplicate=True),
        Input('overlay-tile-click-input', 'value'),
        State('overlay-image-dropdown', 'value'),
        State('overlay-context-store', 'data'),
        prevent_initial_call=True,
    )
    def update_tile_heatmap_store(click_json, image_name, context):
        if not (click_json and context and image_name):
            return dash.no_update, dash.no_update

        try:
            click = json.loads(click_json)
            frac_x, frac_y = float(click['x']), float(click['y'])
        except (TypeError, ValueError, KeyError):
            return dash.no_update, dash.no_update

        try:
            tiles = find_tiles_at_fraction(
                context['base_dir'], context['group'], context['task'], context['model'],
                image_name, frac_x, frac_y,
            )
        except Exception:
            return {'items': [], 'index': 0}, html.Pre(traceback.format_exc(), className='metrics-box')

        if not tiles:
            return {'items': [], 'index': 0}, html.Span('No tile found at that point.', className='status-error')

        heatmaps = collect_tile_heatmaps(
            context['base_dir'], context['group'], context['task'], context['model'], tiles,
        )
        if not heatmaps:
            return {'items': [], 'index': 0}, html.Span(
                f"{len(tiles)} tile(s) found at that point, but no heatmap image files exist for them.",
                className='status-error',
            )

        items = []
        for h in heatmaps:
            try:
                heatmap_image = Image.open(os.path.join(context['base_dir'], h['path']))
                items.append({
                    'label': f"{h['filename']}  (x={h['x']}, y={h['y']})",
                    'src': pil_to_data_uri(heatmap_image),
                })
            except Exception:
                continue

        if not items:
            return {'items': [], 'index': 0}, html.Span(
                f"{len(tiles)} tile(s) found at that point, but their heatmap file(s) could not be opened.",
                className='status-error',
            )

        status = html.Span(f'{len(items)} heatmap(s) at that point', className='status-ok')
        return {'items': items, 'index': 0}, status

    @app.callback(
        Output('tile-heatmap-store', 'data', allow_duplicate=True),
        Input('tile-heatmap-prev-button', 'n_clicks'),
        Input('tile-heatmap-next-button', 'n_clicks'),
        State('tile-heatmap-store', 'data'),
        prevent_initial_call=True,
    )
    def move_tile_heatmap_index(prev_clicks, next_clicks, data):
        items = (data or {}).get('items') or []
        if not items:
            return dash.no_update

        triggered = dash.ctx.triggered_id
        index = (data or {}).get('index', 0)
        if triggered == 'tile-heatmap-prev-button':
            index = (index - 1) % len(items)
        elif triggered == 'tile-heatmap-next-button':
            index = (index + 1) % len(items)
        else:
            return dash.no_update

        return {'items': items, 'index': index}

    @app.callback(
        Output('tile-heatmap-carousel', 'style'),
        Output('tile-heatmap-img', 'src'),
        Output('tile-heatmap-label', 'children'),
        Input('tile-heatmap-store', 'data'),
    )
    def render_tile_heatmap_carousel(data):
        items = (data or {}).get('items') or []
        if not items:
            return {'display': 'none'}, None, ''

        index = (data or {}).get('index', 0) % len(items)
        item = items[index]
        label = f"{item['label']}  ({index + 1} / {len(items)})"
        return {'display': 'block'}, item['src'], label

    @app.callback(
        Output('viewer-status', 'children'),
        Output('viewer-image', 'src'),
        Output('viewer-image', 'data-image-path'),
        Output('viewer-channel-colorbars', 'src'),
        Input('viewer-show-button', 'n_clicks'),
        State('viewer-image-dropdown', 'value'),
        State('viewer-channel-dropdown', 'value'),
        State('viewer-norm-mode', 'value'),
        State('viewer-context-store', 'data'),
    )
    def update_viewer_image(n_clicks, rel_image_path, channel_indices, norm_mode, context):
        if not (n_clicks and context and rel_image_path):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not channel_indices:
            return (
                html.Span('Pick at least one channel.', className='status-error'),
                dash.no_update, dash.no_update, None,
            )

        scale = 0.25

        try:
            pil_img, tif_path, channel_max_values = build_plain_nori_image(
                context['base_dir'], context['group'], context['task'], rel_image_path, scale=scale,
                channel_indices=channel_indices, norm_mode=norm_mode or NORM_MODE_GLOBAL,
            )
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), None, dash.no_update, None

        channel_names = load_channel_names(context['base_dir'], context['group'], context['task'])
        channel_label = ', '.join(channel_names.get(idx, str(idx)) for idx in channel_indices)
        norm_label = 'per-image max' if norm_mode == NORM_MODE_PER_IMAGE else 'calibration file max'
        status = html.Span(
            f'Loaded: {tif_path}  ·  {pil_img.width}×{pil_img.height}px ({round(scale * 100)}%)  ·  '
            f'channels: {channel_label}  ·  normalization: {norm_label}',
            className='status-ok',
        )
        colorbars_img = build_channel_colorbars(channel_indices, channel_names, channel_max_values)
        return status, pil_to_data_uri(pil_img), rel_image_path, pil_to_data_uri(colorbars_img)

    @app.callback(
        Output('region-plot-status', 'children'),
        Output('viewer-regions-container', 'children'),
        Input('viewer-region-input', 'value'),
        Input('viewer-channel-dropdown', 'value'),
        Input('viewer-norm-mode', 'value'),
        State('viewer-context-store', 'data'),
    )
    def update_regions(region_json, channel_indices, norm_mode, context):
        if not context:
            return dash.no_update, dash.no_update

        try:
            entries = json.loads(region_json) if region_json else []
        except (TypeError, ValueError):
            return dash.no_update, dash.no_update

        if not entries:
            return '', []

        region_results = []
        try:
            for entry in entries:
                rel_image_path = entry['image']
                region = {'type': entry.get('type', 'rect'), 'points': entry['points']}
                thumb, px_bbox, protein, lipid, mean_protein, mean_lipid, n_outliers, extra_channels = (
                    build_region_analysis(
                        context['base_dir'], context['group'], context['task'], rel_image_path,
                        region, channel_indices=channel_indices, norm_mode=norm_mode or NORM_MODE_GLOBAL,
                    )
                )
                region_results.append({
                    'thumb': thumb, 'px_bbox': px_bbox, 'image': rel_image_path, 'type': region['type'],
                    'protein': protein, 'lipid': lipid,
                    'mean_protein': mean_protein, 'mean_lipid': mean_lipid, 'n_outliers': n_outliers,
                    'extra_channels': extra_channels,
                })
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), []

        cards = []
        for i, r in enumerate(region_results):
            color = REGION_COLORS[i % len(REGION_COLORS)]
            x0, y0, x1, y1 = r['px_bbox']
            shape_label = 'polygon' if r['type'] == 'polygon' else 'rect'
            cards.append(html.Div(className='region-card', children=[
                html.Div(className='region-card-header', children=[
                    html.Span(str(i + 1), className='region-card-badge', style={'background': color}),
                    html.Div([
                        html.Span(
                            f'{r["image"]}  ·  {shape_label}  ·  x[{x0}:{x1}]  y[{y0}:{y1}]  '
                            f'({r["protein"].size} px, {r["n_outliers"]} outlier px removed)',
                            className='region-card-title',
                        ),
                        html.Div(
                            f'mean protein {r["mean_protein"]:.1f}  ·  mean lipid {r["mean_lipid"]:.1f}',
                            className='status-text', style={'marginTop': '2px'},
                        ),
                    ]),
                    html.Button(
                        '✕', title='Delete this region', className='region-delete-btn',
                        **{'data-region-index': str(i)},
                    ),
                ]),
                html.Div(className='region-thumb-wrap', children=[
                    plot_with_toolbar(
                        html.Img(src=pil_to_data_uri(r['thumb']), className='region-thumb-img'),
                        f'region_{i + 1}_thumbnail',
                    ),
                ]),
            ]))

        if region_results:
            combined_dist = build_combined_distribution_plot(region_results)
            cards.append(html.Div(className='region-card region-combined-card', children=[
                html.Div(className='region-card-header', children=[
                    html.Span('All', className='region-card-badge', style={'background': '#1a1d23'}),
                    html.Span('Combined distribution — all regions overlaid', className='region-card-title'),
                ]),
                plot_with_toolbar(
                    html.Img(
                        src=pil_to_data_uri(combined_dist), className='umap-card-img',
                        style={'width': '62.5%'},
                    ),
                    'all_regions_combined_distribution',
                ),
            ]))

            boxplot_img = build_region_boxplot(region_results)
            cards.append(html.Div(className='region-card', children=[
                html.Div(className='region-card-header', children=[
                    html.Span('All', className='region-card-badge', style={'background': '#1a1d23'}),
                    html.Span('Protein / lipid boxplots by region', className='region-card-title'),
                ]),
                plot_with_toolbar(
                    html.Img(
                        src=pil_to_data_uri(boxplot_img), className='umap-card-img',
                        style={'width': '62.5%'},
                    ),
                    'all_regions_boxplot',
                ),
            ]))

        total_pixels = sum(r['protein'].size for r in region_results)
        status = html.Span(
            f'{len(region_results)} region(s) selected · {total_pixels} px total', className='status-ok',
        )
        return status, cards

    @app.callback(
        Output('base-dir-store', 'data'),
        Output('base-dir-status', 'children'),
        Output('group-dropdown', 'options'),
        Output('group-dropdown', 'value'),
        Input('base-dir-button', 'n_clicks'),
        Input('base-dir-input', 'value'),
    )
    def update_base_dir(n_clicks, base_dir):
        if not base_dir:
            return None, '', [], None

        base_dir = base_dir.strip()
        if not is_valid_base_dir(base_dir):
            status = html.Span(f'"{base_dir}" does not contain an outputs/ folder.', className='status-error')
            return None, status, [], None

        groups = list_groups(base_dir)
        status = html.Span(f'Loaded "{base_dir}" ({len(groups)} group(s) found).', className='status-ok')
        return base_dir, status, [{'label': g, 'value': g} for g in groups], (groups[0] if groups else None)

    @app.callback(
        Output('task-dropdown', 'options'),
        Output('task-dropdown', 'value'),
        Input('group-dropdown', 'value'),
        State('base-dir-store', 'data'),
    )
    def update_tasks(group, base_dir):
        if not (base_dir and group):
            return [], None
        tasks = list_tasks(base_dir, group)
        return [{'label': t, 'value': t} for t in tasks], (tasks[0] if tasks else None)

    @app.callback(
        Output('model-dropdown', 'options'),
        Output('model-dropdown', 'value'),
        Input('task-dropdown', 'value'),
        State('group-dropdown', 'value'),
        State('base-dir-store', 'data'),
    )
    def update_models(task, group, base_dir):
        if not (base_dir and group and task):
            return [], None
        models = list_models(base_dir, group, task)
        return [{'label': m, 'value': m} for m in models], (models[0] if models else None)

    @app.callback(
        Output('umap-store', 'data'),
        Output('metrics-output', 'children'),
        Output('umap-scatter', 'figure'),
        Output('attn-boxplots-img', 'src'),
        Input('show-umap-button', 'n_clicks'),
        State('model-dropdown', 'value'),
        State('group-dropdown', 'value'),
        State('task-dropdown', 'value'),
        State('base-dir-store', 'data'),
    )
    def update_umap(n_clicks, model_name, group, task, base_dir):
        if not (n_clicks and base_dir and group and task and model_name):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        try:
            umap_df = load_umap_df(base_dir, group, task, model_name)
        except Exception:
            return None, f'Error loading UMAP data:\n\n{traceback.format_exc()}', go.Figure(), None

        fig = build_umap_figure(umap_df, group, task, model_name)

        try:
            attn_boxplots_src = build_attention_boxplots(umap_df)
        except Exception:
            attn_boxplots_src = None

        try:
            mae, r2, max_p, max_l = compute_metrics(base_dir, umap_df, group, task)
            metrics_text = (
                f'{model_name}\n\n'
                f'Mean Absolute Error (MAE): {round(mae, 2)}\n'
                f'R² Score: {round(r2, 2)}'
            )
        except Exception:
            metrics_text = f'{model_name}\n\nError computing metrics:\n\n{traceback.format_exc()}'
            max_p = max_l = None

        store_data = {
            'records': umap_df.to_dict('records'),
            'max_p': max_p,
            'max_l': max_l,
        }
        return store_data, metrics_text, fig, attn_boxplots_src

    @app.callback(
        Output('image-panel-output', 'children'),
        Input('umap-scatter', 'clickData'),
        State('umap-store', 'data'),
        State('base-dir-store', 'data'),
    )
    def display_selected_tile(click_data, store_data, base_dir):
        if not click_data or not store_data or not base_dir:
            return html.Div('Click a point in the UMAP plot to inspect a tile.', className='tile-placeholder')

        age, image_name, heatmap_path = click_data['points'][0]['customdata']

        record = next((r for r in store_data['records'] if r['filename'] == image_name), None)
        prediction = record['prediction'] if record else None
        pred_class = record['pred_class'] if record else None

        try:
            heatmap_image = Image.open(os.path.join(base_dir, heatmap_path))
        except FileNotFoundError as exc:
            return html.Div(f'Could not load heatmap image: {exc}', className='status-error')

        prediction_text = round(prediction, 2) if prediction is not None else '?'

        return html.Div([
            html.P([html.Strong('Image: '), image_name], className='tile-meta'),
            html.P([
                html.Strong('Age: '), f'{age}   ',
                html.Strong('Prediction: '), f'{prediction_text}   ',
                html.Strong('Predicted class: '), f'{pred_class}',
            ], className='tile-meta'),
            plot_with_toolbar(
                html.Img(src=pil_to_data_uri(heatmap_image), className='tile-image'),
                os.path.splitext(image_name)[0] + '_heatmap',
            ),
        ])


def start(host='127.0.0.1', port=8050, debug=False, open_browser=True, base_dir=None):
    app = create_app(default_base_dir=base_dir)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f'http://{host}:{port}')).start()

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', default=None, help='Folder containing outputs/ and data/ (default: cwd)')
    args = parser.parse_args()
    start(base_dir=args.base_dir)
