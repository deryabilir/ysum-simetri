import io
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage, stats


st.set_page_config(
    page_title="YSÜM | Görüntü Simetri İncelemesi",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# AYARLAR
# ------------------------------------------------------------

SURUM = "YSÜM-v3.1-Pro"
IMG_N = 200

ALFA = 0.05
KRITER_SAYISI = 3
ALFA_DUZELTILMIS = ALFA / KRITER_SAYISI

MIN_KAPSAMA = 0.50
DUZ_ALANLARI_DISLA = True
YORUM_ESIGI = 0.15
SKOR_YARICAPLARI = (20, 40, 60, 80)
RENK_MIN_ORAN = 0.05

K_DEGERLERI = (2, 5, 6, 8)
K_SECENEKLERI = {
    2: "2 katlı dönel karşılaştırma — 180°",
    5: "5 katlı dönel karşılaştırma — 72°",
    6: "6 katlı dönel karşılaştırma — 60°",
    8: "8 katlı dönel karşılaştırma — 45°",
}

HASSASIYETLER = {
    "Standart": (36, 8),
    "Hassas": (180, 4),
    "Çok hassas": (360, 4),
}

GROUP_NV = "İyi huylu nevüs (nv) etiketi taşıyan görseller"
GROUP_MEL = "Kötü huylu melanom (mel) etiketi taşıyan görseller"
GROUP_NV_CHART = "İyi huylu nevüs (nv)"
GROUP_MEL_CHART = "Kötü huylu melanom (mel)"


# ------------------------------------------------------------
# STREAMLIT SÜRÜM UYUMLULUK YARDIMCILARI
# ------------------------------------------------------------

def ui_image(image, **kwargs):
    """Yeni ve eski Streamlit sürümlerinde görsel gösterir."""
    try:
        st.image(image, width="stretch", **kwargs)
    except TypeError:
        st.image(image, use_container_width=True, **kwargs)


def ui_plot(fig):
    """Yeni ve eski Streamlit sürümlerinde grafik gösterir."""
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)


def ui_dataframe(data, **kwargs):
    """Yeni ve eski Streamlit sürümlerinde tablo gösterir."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


# ------------------------------------------------------------
# TASARIM
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 5% 2%, rgba(84, 190, 179, 0.12), transparent 26%),
            radial-gradient(circle at 95% 0%, rgba(120, 108, 222, 0.10), transparent 25%),
            #f5f8fc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7fa 0%, #f8f5ff 100%);
        border-right: 1px solid #e1e8f1;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(112deg, #155e75 0%, #287f91 42%, #6753a5 100%);
        box-shadow: 0 12px 30px rgba(36, 78, 112, 0.18);
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        font-size: 2.1rem;
        line-height: 1.15;
        margin: 0 0 .45rem 0;
        font-weight: 800;
    }

    .hero p {
        margin: .2rem 0;
        opacity: .94;
        font-size: 1rem;
    }

    .eyebrow {
        font-size: .78rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        opacity: .82;
        font-weight: 700;
        margin-bottom: .55rem;
    }

    .section-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #e5ebf2;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin: .35rem 0 .9rem 0;
        box-shadow: 0 5px 18px rgba(28, 48, 77, .045);
    }

    .result-card {
        background: linear-gradient(110deg, #ffffff 0%, #f5f8ff 100%);
        border: 1px solid #dfe7f3;
        border-left: 5px solid #5267b5;
        border-radius: 14px;
        padding: .9rem 1rem;
        margin: .55rem 0;
        color: #25334a;
    }

    .small-muted {
        color: #5d6b80;
        font-size: .9rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.94);
        border: 1px solid #e4eaf2;
        padding: .8rem 1rem;
        border-radius: 16px;
        box-shadow: 0 5px 16px rgba(28,48,77,.045);
    }

    div[data-testid="stMetricLabel"] {
        color: #526078;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #21375b;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .3rem;
        border-bottom: 1px solid #dce5ef;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,.75);
        border-radius: 12px 12px 0 0;
        padding: .65rem .85rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        color: #4f57a8;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        border: 0;
        color: white;
        background: linear-gradient(100deg, #267c91, #6855a6);
        font-weight: 700;
        min-height: 2.7rem;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        color: white;
        filter: brightness(1.06);
        border: 0;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,.7);
        border-radius: 14px;
        padding: .25rem;
    }

    hr {
        border-color: #e4eaf2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# GÖRSEL VE ÖLÇÜM FONKSİYONLARI
# ------------------------------------------------------------

RESAMPLE = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


def image_from_bytes(data):
    with Image.open(io.BytesIO(data)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resize_rgb(image):
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)

        if arr.ndim == 2:
            pil_image = Image.fromarray(
                np.uint8(np.clip(arr, 0, 255))
            ).convert("RGB")
        else:
            pil_image = Image.fromarray(
                np.uint8(np.clip(arr[..., :3], 0, 255))
            ).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, bytes):
        pil_image = image_from_bytes(image)
    else:
        raise TypeError("Desteklenmeyen görsel biçimi.")

    pil_image = pil_image.resize((IMG_N, IMG_N), RESAMPLE)
    return np.asarray(pil_image, dtype=np.uint8)


def otsu_threshold(gray):
    values = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    total = hist.sum()

    if total <= 0:
        return 127

    levels = np.arange(256, dtype=float)
    total_sum = np.dot(levels, hist)
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        background_weight += hist[threshold]
        background_sum += threshold * hist[threshold]

        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_mean = background_sum / background_weight
        foreground_mean = (total_sum - background_sum) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return int(best_threshold)


def segment_lesion(rgb):
    """Otsu bölütlemesiyle merkezdeki uygun koyu bileşeni seçer."""
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = np.asarray(Image.fromarray(arr).convert("L"), dtype=np.uint8)
    threshold = otsu_threshold(gray)

    dark = gray < threshold
    labels, count = ndimage.label(
        dark,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    height, width = gray.shape
    total_pixels = float(height * width)
    min_area = 0.02 * total_pixels
    max_area = 0.80 * total_pixels

    candidates = []

    for label_id in range(1, count + 1):
        area = int(np.count_nonzero(labels == label_id))
        if min_area <= area <= max_area:
            candidates.append((label_id, area))

    if not candidates:
        return {
            "mask": np.zeros_like(gray, dtype=bool),
            "valid": False,
            "threshold": threshold,
            "area_fraction": 0.0,
        }

    center_label = int(labels[height // 2, width // 2])
    selected_label = None

    if center_label > 0:
        center_area = int(np.count_nonzero(labels == center_label))
        if min_area <= center_area <= max_area:
            selected_label = center_label

    if selected_label is None:
        selected_label = max(candidates, key=lambda item: item[1])[0]

    mask = labels == selected_label
    mask = ndimage.binary_fill_holes(mask)
    area_fraction = float(mask.mean())

    return {
        "mask": mask.astype(bool),
        "valid": bool(0.02 <= area_fraction <= 0.80),
        "threshold": threshold,
        "area_fraction": area_fraction,
    }


def image_center(rgb, center_adjust=False):
    height, width = rgb.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    if center_adjust:
        segmentation = segment_lesion(rgb)

        if segmentation["valid"] and np.any(segmentation["mask"]):
            ys, xs = np.where(segmentation["mask"])
            if len(xs) > 0:
                cx = float(xs.mean())
                cy = float(ys.mean())

    return cx, cy


def sample_binary_pairs(binary, cx, cy, radius_values, theta, rotation_angle):
    height, width = binary.shape
    radii = np.asarray(radius_values, dtype=float)

    x1 = cx + radii * np.cos(theta)
    y1 = cy + radii * np.sin(theta)
    x2 = cx + radii * np.cos(theta + rotation_angle)
    y2 = cy + radii * np.sin(theta + rotation_angle)

    valid = (
        (x1 >= 0)
        & (x1 <= width - 1)
        & (y1 >= 0)
        & (y1 <= height - 1)
        & (x2 >= 0)
        & (x2 <= width - 1)
        & (y2 >= 0)
        & (y2 <= height - 1)
    )

    if not np.any(valid):
        return (
            np.asarray([], dtype=np.uint8),
            np.asarray([], dtype=np.uint8),
            valid,
        )

    x1i = np.rint(x1[valid]).astype(int)
    y1i = np.rint(y1[valid]).astype(int)
    x2i = np.rint(x2[valid]).astype(int)
    y2i = np.rint(y2[valid]).astype(int)

    return binary[y1i, x1i], binary[y2i, x2i], valid


def local_u_map(binary, center, k, baseline_b, window=15):
    binary = np.asarray(binary, dtype=np.uint8)
    height, width = binary.shape
    cx, cy = center

    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx - cx
    dy = yy - cy

    angle = 2.0 * np.pi / float(k)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    source_x = cx + cos_a * dx - sin_a * dy
    source_y = cy + sin_a * dx + cos_a * dy

    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )

    rotated = np.zeros_like(binary, dtype=np.uint8)
    sx = np.rint(source_x[valid]).astype(int)
    sy = np.rint(source_y[valid]).astype(int)
    rotated[valid] = binary[sy, sx]

    coverage = ndimage.uniform_filter(
        valid.astype(float),
        size=window,
        mode="constant",
        cval=0.0,
    )

    matches = ((binary == rotated) & valid).astype(float)
    match_fraction = ndimage.uniform_filter(
        matches,
        size=window,
        mode="constant",
        cval=0.0,
    )

    local_s = np.divide(
        match_fraction,
        coverage,
        out=np.full_like(match_fraction, np.nan, dtype=float),
        where=coverage > 0,
    )

    denominator = 1.0 - baseline_b
    if denominator <= 1e-12:
        return np.full((height, width), np.nan), coverage

    heat = (local_s - baseline_b) / denominator

    pair_min = np.where(valid, np.minimum(binary, rotated), 1).astype(np.uint8)
    pair_max = np.where(valid, np.maximum(binary, rotated), 0).astype(np.uint8)

    local_min = ndimage.minimum_filter(pair_min, size=window, mode="nearest")
    local_max = ndimage.maximum_filter(pair_max, size=window, mode="nearest")
    flat_regions = local_min == local_max

    heat[(coverage < MIN_KAPSAMA) | flat_regions] = np.nan
    return heat, coverage


def gorsel_analiz(
    image,
    k=2,
    n_angles=180,
    step_px=4,
    center_adjust=False,
    with_heat=True,
):
    rgb = resize_rgb(image)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)

    threshold = float(gray.mean())
    binary = (gray >= threshold).astype(np.uint8)

    p = float(binary.mean())
    baseline_b = float(p * p + (1.0 - p) * (1.0 - p))
    center = image_center(rgb, center_adjust=center_adjust)

    step_px = max(int(step_px), 1)

    radius_values = set(range(20, 81, step_px))
    radius_values.update(SKOR_YARICAPLARI)
    radius_values = 
