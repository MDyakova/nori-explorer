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
import webbrowser
from urllib.parse import parse_qs, urlencode

import dash
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from dash import Input, Output, State, dcc, html
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from tifffile import TiffFile


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


def compute_metrics(base_dir, umap_df, group, task):
    max_p, max_l = load_max_values(base_dir, group, task)

    mae = mean_absolute_error(umap_df['age'], umap_df['prediction'])
    r2 = r2_score(umap_df['age'], umap_df['prediction'])
    report = classification_report(umap_df['age'], umap_df['pred_class'])
    conf_matrix = confusion_matrix(umap_df['age'], umap_df['pred_class'])
    return mae, r2, report, conf_matrix, max_p, max_l


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


# --- whole-image prediction overlay -------------------------------------

def image_filter(image):
    """Clip each channel at the median per-tile 99th percentile to remove outliers."""
    all_layers = []
    for layer in range(3):
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
    im = ax.imshow(pred_overlay, cmap='jet', alpha=0.45, vmin=8, vmax=26)
    fig.colorbar(im, ax=ax, label='Prediction')
    ax.set_title(image_name)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}', tif_path, tile_size


def build_plain_nori_image(base_dir, group, task, rel_image_path, scale=1.0):
    """Render the protein/lipid NoRI composite for one whole-slide .tif, resized to `scale` of its original size."""
    tif_path = os.path.join(base_dir, 'data', group, rel_image_path)
    max_p, max_l = load_max_values(base_dir, group, task)

    with TiffFile(tif_path) as tif:
        image = tif.asarray()

    image_nori = np.stack([image[0], image[1], image[0] * 0], axis=0).astype(float)
    filtered_image = image_filter(image_nori)
    filtered_image[0] = filtered_image[0] / max_p
    filtered_image[1] = filtered_image[1] / max_l
    filtered_image = np.clip(filtered_image, 0, 1)

    transformed_image = (filtered_image * 255).transpose((1, 2, 0)).astype(np.uint8)
    pil_img = Image.fromarray(transformed_image)

    width = max(1, round(pil_img.width * scale))
    height = max(1, round(pil_img.height * scale))
    if (width, height) != pil_img.size:
        pil_img = pil_img.resize((width, height), resample=Image.LANCZOS)

    return pil_img, tif_path


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


def build_region_analysis(base_dir, group, task, rel_image_path, bbox, max_thumb_px=220):
    """For a fractional bbox (values in [0, 1], relative to the image's own width/height),
    return a small protein/lipid composite thumbnail of the cropped region plus its
    (outlier-removed, calibrated) protein/lipid pixel values and their means."""
    tif_path = os.path.join(base_dir, 'data', group, rel_image_path)
    max_p, max_l = load_max_values(base_dir, group, task)

    with TiffFile(tif_path) as tif:
        image = tif.asarray()

    h, w = image[0].shape
    x0 = int(np.clip(round(bbox['x0'] * w), 0, w))
    x1 = int(np.clip(round(bbox['x1'] * w), 0, w))
    y0 = int(np.clip(round(bbox['y0'] * h), 0, h))
    y1 = int(np.clip(round(bbox['y1'] * h), 0, h))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError('Selected region is too small to analyze.')

    protein_crop = image[0][y0:y1, x0:x1].astype(float)
    lipid_crop = image[1][y0:y1, x0:x1].astype(float)

    composite = np.stack([protein_crop, lipid_crop, protein_crop * 0], axis=0)
    composite = image_filter(composite)
    composite[0] = composite[0] / max_p
    composite[1] = composite[1] / max_l
    composite = np.clip(composite, 0, 1)
    composite = (composite * 255).transpose((1, 2, 0)).astype(np.uint8)
    thumb = Image.fromarray(composite)
    thumb_scale = min(1.0, max_thumb_px / max(thumb.width, thumb.height))
    if thumb_scale < 1.0:
        thumb = thumb.resize(
            (max(1, round(thumb.width * thumb_scale)), max(1, round(thumb.height * thumb_scale))),
            resample=Image.LANCZOS,
        )

    protein_raw = protein_crop.ravel() * PROTEIN_CALIBRATION_K
    lipid_raw = lipid_crop.ravel() * LIPID_CALIBRATION_K
    protein, lipid = remove_outliers(protein_raw, lipid_raw)
    n_outliers = protein_raw.size - protein.size
    mean_protein = float(protein.mean())
    mean_lipid = float(lipid.mean())

    return thumb, (x0, y0, x1, y1), protein, lipid, mean_protein, mean_lipid, n_outliers


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
    ax.set_xlabel('Protein')
    ax.set_ylabel('Lipid')
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

    cards = [build_model_card(base_dir, group, task, model_name) for model_name in model_names]

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
                html.Img(src=combined_src, className='umap-card-img'), 'predictions_by_age_and_sample',
            ) if combined_src
            else html.Pre(combined_error or 'No data available.', className='metrics-box'),
        ]))

    return html.Div(className='app-shell', children=[
        html.A('← Back to explorer', href='/', className='btn-outline'),
        html.Div(className='app-header', children=[
            html.H2(f'All model plots — {group} / {task}'),
            html.P(f'{len(model_names)} model(s) found.'),
        ]),
        html.Div(className='umap-grid', children=cards),
        *combined_section,
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
                html.Div([
                    html.Label('Figure scale (%)', className='field-label'),
                    dcc.Input(
                        id='overlay-scale-input', type='number', value=100, min=10, max=300,
                        className='dash-input',
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
            plot_with_toolbar(
                html.Img(id='overlay-image', className='umap-card-img', style={'marginTop': '12px'}),
                'whole_image_overlay',
            ),
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
                    html.Label('Size (% of original)', className='field-label'),
                    dcc.Input(
                        id='viewer-scale-input', type='number', value=25, min=1, max=500,
                        className='dash-input',
                    ),
                ]),
            ]),
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
                    'Select region', id='viewer-select-mode-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Button(
                    'Clear regions', id='viewer-clear-regions-button', n_clicks=0,
                    className='btn-outline', style={'flex': '0 0 auto'},
                ),
                html.Div(
                    'Drag on the image to pick a region (repeat to pick several); each '
                    "one's protein/lipid distribution is plotted below.",
                    className='status-text', style={'marginTop': 0},
                ),
            ]),
            dcc.Input(id='viewer-region-input', type='text', value='[]', style={'display': 'none'}),
            html.Div(className='plot-wrap', style={'marginTop': '12px'}, children=[
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
                        html.Img(id='viewer-image', className='zoom-image'),
                    ]),
                ]),
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
        Output('overlay-status', 'children'),
        Output('overlay-image', 'src'),
        Input('overlay-show-button', 'n_clicks'),
        State('overlay-image-dropdown', 'value'),
        State('overlay-scale-input', 'value'),
        State('overlay-context-store', 'data'),
    )
    def update_overlay(n_clicks, image_name, scale, context):
        if not (n_clicks and context and image_name):
            return dash.no_update, dash.no_update

        try:
            scale = float(scale) / 100
        except (TypeError, ValueError):
            scale = 1.0
        fig_width = ORIGINAL_OVERLAY_FIG_WIDTH * scale
        fig_height = ORIGINAL_OVERLAY_FIG_HEIGHT * scale

        try:
            img_src, tif_path, tile_size = build_whole_image_overlay(
                context['base_dir'], context['group'], context['task'], context['model'], image_name,
                fig_width=fig_width, fig_height=fig_height,
            )
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), None

        status = html.Span(
            f'Loaded: {tif_path}  ·  Tile size: {tile_size}px (auto-detected)', className='status-ok',
        )
        return status, img_src

    @app.callback(
        Output('viewer-status', 'children'),
        Output('viewer-image', 'src'),
        Input('viewer-show-button', 'n_clicks'),
        State('viewer-image-dropdown', 'value'),
        State('viewer-scale-input', 'value'),
        State('viewer-context-store', 'data'),
    )
    def update_viewer_image(n_clicks, rel_image_path, scale, context):
        if not (n_clicks and context and rel_image_path):
            return dash.no_update, dash.no_update

        try:
            scale = float(scale) / 100
        except (TypeError, ValueError):
            scale = 0.25

        try:
            pil_img, tif_path = build_plain_nori_image(
                context['base_dir'], context['group'], context['task'], rel_image_path, scale=scale,
            )
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), None

        status = html.Span(
            f'Loaded: {tif_path}  ·  {pil_img.width}×{pil_img.height}px ({round(scale * 100)}%)',
            className='status-ok',
        )
        return status, pil_to_data_uri(pil_img)

    @app.callback(
        Output('region-plot-status', 'children'),
        Output('viewer-regions-container', 'children'),
        Input('viewer-region-input', 'value'),
        State('viewer-image-dropdown', 'value'),
        State('viewer-context-store', 'data'),
    )
    def update_regions(region_json, rel_image_path, context):
        if not (rel_image_path and context):
            return dash.no_update, dash.no_update

        try:
            bboxes = json.loads(region_json) if region_json else []
        except (TypeError, ValueError):
            return dash.no_update, dash.no_update

        if not bboxes:
            return '', []

        region_results = []
        try:
            for bbox in bboxes:
                thumb, px_bbox, protein, lipid, mean_protein, mean_lipid, n_outliers = (
                    build_region_analysis(
                        context['base_dir'], context['group'], context['task'], rel_image_path, bbox,
                    )
                )
                region_results.append({
                    'thumb': thumb, 'px_bbox': px_bbox,
                    'protein': protein, 'lipid': lipid,
                    'mean_protein': mean_protein, 'mean_lipid': mean_lipid, 'n_outliers': n_outliers,
                })
        except Exception:
            return html.Pre(traceback.format_exc(), className='metrics-box'), []

        cards = []
        for i, r in enumerate(region_results):
            color = REGION_COLORS[i % len(REGION_COLORS)]
            x0, y0, x1, y1 = r['px_bbox']
            cards.append(html.Div(className='region-card', children=[
                html.Div(className='region-card-header', children=[
                    html.Span(str(i + 1), className='region-card-badge', style={'background': color}),
                    html.Div([
                        html.Span(
                            f'x[{x0}:{x1}]  y[{y0}:{y1}]  ({r["protein"].size} px, '
                            f'{r["n_outliers"]} outlier px removed)',
                            className='region-card-title',
                        ),
                        html.Div(
                            f'mean protein {r["mean_protein"]:.1f}  ·  mean lipid {r["mean_lipid"]:.1f}',
                            className='status-text', style={'marginTop': '2px'},
                        ),
                    ]),
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
            mae, r2, report, conf_matrix, max_p, max_l = compute_metrics(base_dir, umap_df, group, task)
            metrics_text = (
                f'{model_name}\n\n'
                f'Mean Absolute Error (MAE): {round(mae, 2)}\n'
                f'R² Score: {round(r2, 2)}\n\n'
                f'{report}\n'
                f'Confusion matrix:\n{conf_matrix}'
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
