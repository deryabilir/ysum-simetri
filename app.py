# YSÜM — Yerel Simetri Üstünlük Modeli
# v3.3 — Ana A/B/C protokolü korunmuştur.
# Yeni ayna-asimetrisi analizleri keşifseldir ve ana sonuçlardan ayrıdır.

import hashlib
import html
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from scipy import ndimage, stats


# ---------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="YSÜM | Görüntü Simetri İncelemesi",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SURUM = "YSÜM-v3.3"
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


# ---------------------------------------------------------------------
# GÖRSEL TASARIM
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# GÖRSEL VE ÖLÇÜM YARDIMCILARI
# ---------------------------------------------------------------------

_RESAMPLE = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


def image_from_bytes(data):
    """Bayt verisini EXIF yönü düzeltilmiş RGB PIL görseline çevirir."""
    if not data:
        raise ValueError("Yüklenen dosya boş.")

    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()

        with Image.open(io.BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Dosya geçerli veya desteklenen bir görsel değil: {exc}"
        ) from exc


def resize_rgb(image):
    """Görseli RGB biçiminde 200×200 boyutuna getirir."""
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)

        if arr.ndim == 2:
            pil_image = Image.fromarray(
                np.uint8(np.clip(arr, 0, 255))
            ).convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pil_image = Image.fromarray(
                np.uint8(np.clip(arr, 0, 255))
            ).convert("RGB")
        else:
            raise ValueError(
                "Görsel dizisi H×W, H×W×3 veya H×W×4 biçiminde olmalıdır."
            )
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, bytes):
        pil_image = image_from_bytes(image)
    else:
        raise TypeError("Desteklenmeyen görsel biçimi.")

    pil_image = pil_image.resize((IMG_N, IMG_N), _RESAMPLE)
    return np.asarray(pil_image, dtype=np.uint8)


def otsu_threshold(gray):
    """Gri tonlu görüntü için Otsu eşik değerini hesaplar."""
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
        between_variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = threshold

    return int(best_threshold)


def segment_lesion(rgb):
    """
    Otsu eşiğiyle koyu bileşenleri bulur.
    Görüntü merkezindeki uygun bileşeni; yoksa en büyük uygun bileşeni seçer.
    """
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

    component_sizes = np.bincount(labels.ravel(), minlength=count + 1)
    candidate_ids = [
        label_id
        for label_id in range(1, count + 1)
        if min_area <= component_sizes[label_id] <= max_area
    ]

    if not candidate_ids:
        return {
            "mask": np.zeros_like(gray, dtype=bool),
            "valid": False,
            "threshold": threshold,
            "area_fraction": 0.0,
        }

    center_label = int(labels[height // 2, width // 2])
    selected_label = None

    if (
        center_label > 0
        and min_area <= component_sizes[center_label] <= max_area
    ):
        selected_label = center_label
    else:
        selected_label = max(
            candidate_ids,
            key=lambda label_id: component_sizes[label_id],
        )

    lesion = labels == selected_label
    lesion = ndimage.binary_fill_holes(lesion)
    area_fraction = float(lesion.mean())
    valid = 0.02 <= area_fraction <= 0.80

    return {
        "mask": lesion.astype(bool),
        "valid": bool(valid),
        "threshold": threshold,
        "area_fraction": area_fraction,
    }


def image_center(rgb, center_adjust=False):
    """Varsayılan merkez geometriktir; isteğe bağlı koyu bileşene göre ayarlanır."""
    height, width = rgb.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    if center_adjust:
        seg = segment_lesion(rgb)
        if seg["valid"] and np.any(seg["mask"]):
            ys, xs = np.where(seg["mask"])
            if len(xs) > 0:
                cx = float(xs.mean())
                cy = float(ys.mean())

    return cx, cy


def _sample_binary_pairs(binary, cx, cy, radius_values, theta, rotation_angle):
    """Bir açısal doğrultudaki dönel piksel çiftlerini örnekler."""
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

    first = binary[y1i, x1i]
    second = binary[y2i, x2i]
    return first, second, valid


def local_u_map(binary, center, k, baseline_b, window=15):
    """Yerel dönel eşleşme oranından görsel amaçlı U ısı haritası üretir."""
    binary = np.asarray(binary, dtype=np.uint8)

    if binary.ndim != 2:
        raise ValueError("Isı haritası için ikili görüntü iki boyutlu olmalıdır.")
    if not isinstance(k, (int, np.integer)) or int(k) < 2:
        raise ValueError("Dönel karşılaştırma mertebesi k en az 2 olmalıdır.")

    window = max(1, int(window))
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
        return np.full((height, width), np.nan, dtype=float), coverage

    heat = (local_s - baseline_b) / denominator

    pair_min = np.where(valid, np.minimum(binary, rotated), 1).astype(np.uint8)
    pair_max = np.where(valid, np.maximum(binary, rotated), 0).astype(np.uint8)
    local_min = ndimage.minimum_filter(
        pair_min,
        size=window,
        mode="nearest",
    )
    local_max = ndimage.maximum_filter(
        pair_max,
        size=window,
        mode="nearest",
    )
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
    """
    Ana A ölçümü:
    - Gri tonlama ve 200×200 boyutlandırma
    - Ortalama parlaklık eşiğiyle ikileştirme
    - k katlı dönel piksel çiftlerinin eşleşme oranı S
    - Her görüntü için kendi B(p) şans düzeyi
    - U = (S - B(p)) / (1 - B(p))
    """
    if not isinstance(k, (int, np.integer)) or int(k) < 2:
        raise ValueError("Dönel karşılaştırma mertebesi k en az 2 olmalıdır.")
    if not isinstance(n_angles, (int, np.integer)) or int(n_angles) < 1:
        raise ValueError("Açı örneği sayısı en az 1 olmalıdır.")
    if not isinstance(step_px, (int, np.integer)) or int(step_px) < 1:
        raise ValueError("Radyal örnekleme adımı en az 1 piksel olmalıdır.")

    rgb = resize_rgb(image)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)

    threshold = float(gray.mean())
    binary = (gray >= threshold).astype(np.uint8)

    p = float(binary.mean())
    baseline_b = float(p * p + (1.0 - p) * (1.0 - p))
    center = image_center(rgb, center_adjust=center_adjust)

    radius_values = set(range(20, 81, int(step_px)))
    radius_values.update(SKOR_YARICAPLARI)
    radius_values = sorted(
        radius
        for radius in radius_values
        if 0 < radius <= 80
    )

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    rotation_angle = 2.0 * np.pi / float(k)

    equal_count = 0
    compared_count = 0
    valid_pair_count = 0
    possible_pair_count = int(len(angles) * len(radius_values))
    accepted_profiles = 0
    excluded_flat_profiles = 0

    for theta in angles:
        first, second, valid = _sample_binary_pairs(
            binary,
            center[0],
            center[1],
            radius_values,
            theta,
            rotation_angle,
        )

        valid_count = int(np.count_nonzero(valid))
        valid_pair_count += valid_count

        coverage_at_angle = (
            valid_count / float(len(radius_values))
            if len(radius_values) > 0
            else 0.0
        )

        if coverage_at_angle < MIN_KAPSAMA or len(first) == 0:
            continue

        if DUZ_ALANLARI_DISLA:
            combined = np.concatenate([first, second])
            if combined.size > 0 and np.all(combined == combined[0]):
                excluded_flat_profiles += 1
                continue

        accepted_profiles += 1
        equal_count += int(np.count_nonzero(first == second))
        compared_count += int(len(first))

    score_s = (
        float(equal_count / compared_count)
        if compared_count > 0
        else float("nan")
    )

    if 1.0 - baseline_b > 1e-12 and np.isfinite(score_s):
        score_u = float((score_s - baseline_b) / (1.0 - baseline_b))
    else:
        score_u = float("nan")

    overall_coverage = (
        float(valid_pair_count / possible_pair_count)
        if possible_pair_count > 0
        else 0.0
    )

    if with_heat:
        heat, heat_coverage = local_u_map(binary, center, k, baseline_b)
    else:
        heat = None
        heat_coverage = None

    return {
        "rgb": rgb,
        "gray": gray,
        "binary": binary,
        "threshold": threshold,
        "p": p,
        "B_chance": baseline_b,
        "S": score_s,
        "U": score_u,
        "k": int(k),
        "n_angles": int(n_angles),
        "step_px": int(step_px),
        "center": center,
        "coverage": overall_coverage,
        "accepted_profiles": int(accepted_profiles),
        "excluded_flat_profiles": int(excluded_flat_profiles),
        "compared_pairs": int(compared_count),
        "heatmap": heat,
        "heat_coverage": heat_coverage,
    }


def mirror_mask_asymmetry(mask, axis):
    """
    Otsu maskesinin kendi ağırlık merkezinden yansıtılmasıyla şekil
    örtüşmezliğini hesaplar: XOR / union.

    axis='vertical': sol-sağ yansıması, dikey eksen.
    axis='horizontal': üst-alt yansıması, yatay eksen.

    Bu, renk asimetrisi veya ABCD kuralının tam uygulaması değildir.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        return float("nan")

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan")

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))

    if axis == "vertical":
        reflected_y = ys.copy()
        reflected_x = np.rint(2.0 * cx - xs).astype(int)
    elif axis == "horizontal":
        reflected_y = np.rint(2.0 * cy - ys).astype(int)
        reflected_x = xs.copy()
    else:
        raise ValueError("axis 'vertical' veya 'horizontal' olmalıdır.")

    y_min = int(min(np.min(ys), np.min(reflected_y)))
    y_max = int(max(np.max(ys), np.max(reflected_y)))
    x_min = int(min(np.min(xs), np.min(reflected_x)))
    x_max = int(max(np.max(xs), np.max(reflected_x)))

    height = y_max - y_min + 1
    width = x_max - x_min + 1

    original = np.zeros((height, width), dtype=bool)
    reflected = np.zeros((height, width), dtype=bool)

    original[ys - y_min, xs - x_min] = True
    reflected[reflected_y - y_min, reflected_x - x_min] = True

    union_count = int(np.count_nonzero(original | reflected))
    if union_count == 0:
        return float("nan")

    xor_count = int(np.count_nonzero(original ^ reflected))
    return float(xor_count / union_count)


def lesion_features(rgb):
    """B, C ve keşifsel maske ayna-asimetrisi ölçümlerini hesaplar."""
    seg = segment_lesion(rgb)

    invalid_result = {
        "segmentation_valid": False,
        "area_fraction": seg["area_fraction"],
        "B_edge": float("nan"),
        "C_colors": float("nan"),
        "color_names": "",
        "mirror_asym_lr": float("nan"),
        "mirror_asym_tb": float("nan"),
    }

    if not seg["valid"] or not np.any(seg["mask"]):
        return invalid_result

    mask = seg["mask"]
    area = float(np.count_nonzero(mask))

    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = mask & (~eroded)
    perimeter = float(np.count_nonzero(boundary))

    edge_irregularity = (
        (perimeter * perimeter) / (4.0 * math.pi * area)
        if area > 0
        else float("nan")
    )

    palette = np.asarray(
        [
            [25, 25, 25],       # Siyah
            [85, 50, 35],       # Koyu kahverengi
            [160, 105, 65],     # Açık kahverengi
            [190, 45, 45],      # Kırmızı
            [95, 115, 145],     # Mavi-gri
            [238, 235, 225],    # Beyaz
        ],
        dtype=float,
    )
    palette_names = [
        "Siyah",
        "Koyu kahverengi",
        "Açık kahverengi",
        "Kırmızı",
        "Mavi-gri",
        "Beyaz",
    ]

    pixels = np.asarray(rgb, dtype=float)[mask]
    if pixels.size == 0:
        return invalid_result

    distances = np.sum(
        (pixels[:, None, :] - palette[None, :, :]) ** 2,
        axis=2,
    )
    class_ids = np.argmin(distances, axis=1)
    counts = np.bincount(class_ids, minlength=len(palette_names))
    fractions = counts / float(max(counts.sum(), 1))

    present = [
        palette_names[index]
        for index, fraction in enumerate(fractions)
        if fraction >= RENK_MIN_ORAN
    ]

    return {
        "segmentation_valid": True,
        "area_fraction": float(seg["area_fraction"]),
        "B_edge": float(edge_irregularity),
        "C_colors": float(len(present)),
        "color_names": ", ".join(present),
        "mirror_asym_lr": mirror_mask_asymmetry(mask, "vertical"),
        "mirror_asym_tb": mirror_mask_asymmetry(mask, "horizontal"),
    }


# ---------------------------------------------------------------------
# SENTETİK ÖRNEKLER VE ŞEMATİK GRAFİKLER
# ---------------------------------------------------------------------

SENTETIK_ORNEKLER = {
    "Yuvarlak sentetik motif": ("circle", 2),
    "İkili dönel sentetik motif": ("twofold", 2),
    "Sentetik beş katlı motif (cos(5θ))": ("fivefold", 5),
    "Asimetrik sentetik motif": ("asymmetric", 2),
    "Gürültülü kenarlı sentetik beş katlı motif": ("noisy_fivefold", 5),
}


def sentetik_gorsel(kind="fivefold", n=256, disturbance=0, seed=42):
    """Kontrollü matematiksel sentetik motif üretir; Penrose döşemesi değildir."""
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - (n - 1) / 2.0) / (n / 2.0)
    y = (yy - (n - 1) / 2.0) / (n / 2.0)

    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    if kind == "circle":
        boundary = np.full_like(radius, 0.62)
    elif kind == "twofold":
        boundary = 0.59 * (1.0 + 0.13 * np.cos(2.0 * theta))
    elif kind == "fivefold":
        boundary = 0.60 * (1.0 + 0.14 * np.cos(5.0 * theta))
    elif kind == "noisy_fivefold":
        boundary = 0.60 * (
            1.0
            + 0.14 * np.cos(5.0 * theta)
            + 0.025 * np.sin(17.0 * theta)
        )
    elif kind == "asymmetric":
        boundary = 0.57 * (
            1.0
            + 0.14 * np.cos(theta)
            + 0.07 * np.sin(3.0 * theta)
        )
    else:
        boundary = np.full_like(radius, 0.60)

    if disturbance:
        angular_distance = np.angle(np.exp(1j * (theta - 0.55)))
        notch = np.exp(-0.5 * (angular_distance / 0.18) ** 2)
        boundary = boundary - (float(disturbance) / 100.0) * 0.34 * notch

    inside = radius <= boundary
    texture = rng.normal(0.0, 2.0, size=(n, n))

    background = np.zeros((n, n, 3), dtype=np.uint8)
    background[:] = [245, 247, 244]

    foreground_level = np.clip(
        77.0 + 13.0 * (radius / max(float(radius.max()), 1e-6)) + texture,
        25,
        150,
    ).astype(np.uint8)

    background[..., 0] = np.where(inside, foreground_level, 245)
    background[..., 1] = np.where(inside, foreground_level - 6, 247)
    background[..., 2] = np.where(inside, foreground_level + 4, 244)

    return Image.fromarray(background)


def fig_penrose_baseline():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    p_ref = 1.0 / phi
    b_ref = p_ref**2 + (1.0 - p_ref) ** 2

    p = np.linspace(0.0, 1.0, 401)
    b = p**2 + (1.0 - p) ** 2

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.plot(
        p,
        b,
        color="#365fa7",
        linewidth=3,
        label="B(p) = p² + (1−p)²",
    )
    ax.scatter([p_ref], [b_ref], color="#e26b52", s=90, zorder=5)
    ax.axvline(p_ref, color="#e26b52", linestyle="--", alpha=0.7)
    ax.axhline(0.5, color="#7a8798", linestyle=":", alpha=0.8)
    ax.annotate(
        f"p = 1/φ ≈ {p_ref:.3f}\nB(p) ≈ {b_ref:.3f}",
        xy=(p_ref, b_ref),
        xytext=(0.70, 0.69),
        arrowprops={"arrowstyle": "->", "color": "#e26b52"},
        fontsize=10,
    )
    ax.set_xlabel("Açık piksel oranı p")
    ax.set_ylabel("Rastlantısal eşleşme düzeyi B(p)")
    ax.set_title("Teorik referans eğrisi")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.03)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_schematic_diffraction():
    x = np.linspace(-5, 5, 420)
    y = np.linspace(-5, 5, 420)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)

    radial = (np.cos(2.1 * rr) ** 2) * np.exp(-0.16 * rr)
    angular = 0.25 + 0.75 * (np.cos(5.0 * theta) ** 2)
    intensity = radial * angular

    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    ax.imshow(
        intensity,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="magma",
    )
    ax.set_title("Şematik beş katlı açısal örüntü")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_fivefold_rosette():
    angles = np.linspace(-np.pi / 2.0, 3.0 * np.pi / 2.0, 11)
    radii = np.where(np.arange(11) % 2 == 0, 1.0, 0.43)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    ax.fill(
        x,
        y,
        color="#68b7aa",
        alpha=0.82,
        edgecolor="#205e70",
        linewidth=2,
    )
    ax.plot(x, y, color="#205e70", linewidth=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Genel beş katlı yıldız şeması")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_girih_schematic():
    angles = np.linspace(0, 2.0 * np.pi, 10, endpoint=False) + np.pi / 10.0
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    for step, color, width in [
        (1, "#8d67a8", 1.2),
        (3, "#347f91", 1.8),
        (4, "#dc8d4c", 1.2),
    ]:
        for index in range(10):
            other = (index + step) % 10
            ax.plot(
                [points[index, 0], points[other, 0]],
                [points[index, 1], points[other, 1]],
                color=color,
                linewidth=width,
                alpha=0.82,
            )

    ax.scatter(points[:, 0], points[:, 1], s=16, color="#244c67", zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_title("Girih esintili şematik on katlı motif")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_heat_overlay(rgb, heat, title, show_focus=True):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(rgb)

    if heat is not None and np.isfinite(heat).any():
        masked = np.ma.masked_invalid(heat)
        image = ax.imshow(
            masked,
            cmap="coolwarm",
            alpha=0.58,
            vmin=-1.0,
            vmax=1.0,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Yerel U değeri")

        if show_focus:
            locations = np.argwhere(np.isfinite(heat))
            if len(locations) > 0:
                values = heat[np.isfinite(heat)]
                max_point = locations[int(np.argmax(values))]
                min_point = locations[int(np.argmin(values))]
                ax.scatter(
                    [max_point[1], min_point[1]],
                    [max_point[0], min_point[0]],
                    c=["#22a06b", "#d64545"],
                    s=55,
                    edgecolors="white",
                    linewidths=1.1,
                )
                ax.text(
                    max_point[1] + 3,
                    max_point[0] - 3,
                    "yüksek",
                    color="white",
                    fontsize=8,
                    weight="bold",
                )
                ax.text(
                    min_point[1] + 3,
                    min_point[0] + 10,
                    "düşük",
                    color="white",
                    fontsize=8,
                    weight="bold",
                )
    else:
        ax.text(
            0.5,
            0.5,
            "Gösterilecek yerel değer yok",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            bbox={"facecolor": "#334155", "alpha": 0.75, "pad": 6},
        )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _draw_manual_box(ax, values, position, color):
    """Matplotlib boxplot API'sine bağlı olmayan, sürüm uyumlu kutu grafiği."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inliers = values[(values >= lower_fence) & (values <= upper_fence)]

    whisker_low = (
        float(np.min(inliers))
        if len(inliers)
        else float(np.min(values))
    )
    whisker_high = (
        float(np.max(inliers))
        if len(inliers)
        else float(np.max(values))
    )

    half_width = 0.25
    left = position - half_width
    right = position + half_width
    cap_width = 0.12

    ax.plot(
        [position, position],
        [whisker_low, q1],
        color="#334155",
        linewidth=1.4,
    )
    ax.plot(
        [position, position],
        [q3, whisker_high],
        color="#334155",
        linewidth=1.4,
    )
    ax.plot(
        [position - cap_width, position + cap_width],
        [whisker_low, whisker_low],
        color="#334155",
        linewidth=1.4,
    )
    ax.plot(
        [position - cap_width, position + cap_width],
        [whisker_high, whisker_high],
        color="#334155",
        linewidth=1.4,
    )
    ax.fill_between(
        [left, right],
        [q1, q1],
        [q3, q3],
        color=color,
        alpha=0.72,
        zorder=2,
    )
    ax.plot(
        [left, right, right, left, left],
        [q1, q1, q3, q3, q1],
        color="#334155",
        linewidth=1.3,
        zorder=3,
    )
    ax.plot(
        [left, right],
        [median, median],
        color="#26364c",
        linewidth=2.2,
        zorder=4,
    )

    outliers = values[(values < lower_fence) | (values > upper_fence)]
    if len(outliers) > 0:
        jitter = np.linspace(-0.07, 0.07, len(outliers))
        ax.scatter(
            position + jitter,
            outliers,
            s=15,
            color=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
        )


def _batch_boxplot(data, specs, figsize):
    fig, axes = plt.subplots(1, len(specs), figsize=figsize)
    axes = np.atleast_1d(axes)

    groups = [
        (GROUP_NV, GROUP_NV_CHART, "#55a9a2"),
        (GROUP_MEL, GROUP_MEL_CHART, "#8b72bd"),
    ]

    for ax, (title, column) in zip(axes, specs):
        plot_values = []

        for group_name, label, color in groups:
            if column in data.columns:
                source = data.loc[data["grup"] == group_name, column]
            else:
                source = []

            values = finite_values(source)
            plot_values.append((label, color, values))

        if any(len(values) == 0 for _, _, values in plot_values):
            ax.text(
                0.5,
                0.5,
                "Yeterli değer yok",
                ha="center",
                va="center",
            )
            ax.set_title(title)
            ax.axis("off")
            continue

        for position, (_, color, values) in enumerate(plot_values, start=1):
            _draw_manual_box(ax, values, position, color)

        ax.set_title(title)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [GROUP_NV_CHART, GROUP_MEL_CHART],
            rotation=12,
            ha="right",
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.22)
        ax.margins(y=0.12)

    fig.tight_layout()
    return fig


def fig_batch_boxplots(data):
    return _batch_boxplot(
        data,
        [
            ("A — Dönel simetri U", "A_U"),
            ("B — Kenar düzensizliği", "B_edge"),
            ("C — Renk sınıfı sayısı", "C_colors"),
        ],
        figsize=(15, 4.8),
    )


def fig_mirror_boxplots(data):
    return _batch_boxplot(
        data,
        [
            ("Keşifsel sol-sağ ayna asimetrisi", "mirror_asym_lr"),
            ("Keşifsel üst-alt ayna asimetrisi", "mirror_asym_tb"),
        ],
        figsize=(10.5, 4.8),
    )


# ---------------------------------------------------------------------
# İSTATİSTİKSEL YARDIMCILAR
# ---------------------------------------------------------------------

def finite_values(values):
    """Sayısal ve sonlu değerleri güvenli biçimde numpy dizisine çevirir."""
    try:
        series = pd.Series(values)
    except Exception:
        series = pd.Series([])

    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def fmt_number(value, digits=4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not np.isfinite(number):
        return "—"

    return f"{number:.{digits}f}"


def fmt_p(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not np.isfinite(number):
        return "—"
    if number < 0.0001:
        return "<0.0001"

    return f"{number:.4f}"


def fmt_ci(low, high, digits=4):
    if not np.isfinite(low) or not np.isfinite(high):
        return "—"
    return f"[{fmt_number(low, digits)}, {fmt_number(high, digits)}]"


def welch_directional(expected_high, expected_low):
    """
    Önceden belirlenmiş yön için Welch t testi.
    Beklenen yüksek grup ortalaması − düşük grup ortalaması farkı test edilir.
    """
    x = finite_values(expected_high)
    y = finite_values(expected_low)

    n1 = len(x)
    n2 = len(y)

    result = {
        "n_high": n1,
        "n_low": n2,
        "mean_high": float(np.mean(x)) if n1 else float("nan"),
        "mean_low": float(np.mean(y)) if n2 else float("nan"),
        "difference": float("nan"),
        "t": float("nan"),
        "df": float("nan"),
        "p_one": float("nan"),
        "p_two": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "cohen_d": float("nan"),
    }

    if n1 < 2 or n2 < 2:
        return result

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    difference = mean_x - mean_y

    term_x = var_x / n1
    term_y = var_y / n2
    se_squared = term_x + term_y

    result["difference"] = difference

    # İki grupta da varyans sıfırsa örneklemden standart hata kestirilemez.
    if se_squared <= 0:
        return result

    standard_error = math.sqrt(se_squared)
    t_value = difference / standard_error

    denominator = 0.0
    if n1 > 1:
        denominator += (term_x * term_x) / (n1 - 1)
    if n2 > 1:
        denominator += (term_y * term_y) / (n2 - 1)

    df = (
        se_squared * se_squared / denominator
        if denominator > 0
        else float("inf")
    )

    if np.isfinite(df):
        p_one = float(stats.t.sf(t_value, df))
        p_two = float(2.0 * stats.t.sf(abs(t_value), df))
        critical = float(stats.t.ppf(0.975, df))
    else:
        p_one = float(stats.norm.sf(t_value))
        p_two = float(2.0 * stats.norm.sf(abs(t_value)))
        critical = float(stats.norm.ppf(0.975))

    ci_low = difference - critical * standard_error
    ci_high = difference + critical * standard_error

    pooled_denominator = n1 + n2 - 2
    if pooled_denominator > 0:
        pooled_variance = (
            ((n1 - 1) * var_x + (n2 - 1) * var_y)
            / pooled_denominator
        )
        pooled_sd = math.sqrt(max(pooled_variance, 0.0))
        cohen_d = difference / pooled_sd if pooled_sd > 0 else float("nan")
    else:
        cohen_d = float("nan")

    result.update(
        {
            "t": float(t_value),
            "df": float(df),
            "p_one": float(p_one),
            "p_two": float(p_two),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "cohen_d": float(cohen_d),
        }
    )
    return result


def auc_delong_ci(expected_high, expected_low, confidence=0.95):
    """
    AUC ve DeLong tipi asimptotik güven aralığı.

    AUC, beklenen yüksek gruptan rastgele bir değerin düşük gruptan
    rastgele bir değerden yüksek olma olasılığıdır; eşitlikler yarım sayılır.
    """
    x = finite_values(expected_high)
    y = finite_values(expected_low)

    n_high = len(x)
    n_low = len(y)

    result = {
        "auc": float("nan"),
        "auc_ci_low": float("nan"),
        "auc_ci_high": float("nan"),
        "auc_se": float("nan"),
        "auc_n_high": n_high,
        "auc_n_low": n_low,
    }

    if n_high == 0 or n_low == 0:
        return result

    sorted_y = np.sort(y)
    y_left = np.searchsorted(sorted_y, x, side="left")
    y_right = np.searchsorted(sorted_y, x, side="right")
    v10 = (y_left + 0.5 * (y_right - y_left)) / float(n_low)

    sorted_x = np.sort(x)
    x_left = np.searchsorted(sorted_x, y, side="left")
    x_right = np.searchsorted(sorted_x, y, side="right")
    v01 = (
        (n_high - x_right) + 0.5 * (x_right - x_left)
    ) / float(n_high)

    auc = float(0.5 * (np.mean(v10) + np.mean(v01)))
    result["auc"] = auc

    if n_high < 2 or n_low < 2:
        return result

    variance = (
        float(np.var(v10, ddof=1)) / n_high
        + float(np.var(v01, ddof=1)) / n_low
    )
    standard_error = math.sqrt(max(variance, 0.0))
    z_value = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))

    result.update(
        {
            "auc_se": standard_error,
            "auc_ci_low": max(0.0, auc - z_value * standard_error),
            "auc_ci_high": min(1.0, auc + z_value * standard_error),
        }
    )
    return result


def mann_whitney_two_sided(sample_mel, sample_nv):
    """İki yönlü Mann–Whitney U testi; ilk örnek mel grubudur."""
    mel = finite_values(sample_mel)
    nv = finite_values(sample_nv)

    result = {
        "U": float("nan"),
        "p_two": float("nan"),
        "n_mel": len(mel),
        "n_nv": len(nv),
    }

    if len(mel) < 2 or len(nv) < 2:
        return result

    try:
        test = stats.mannwhitneyu(
            mel,
            nv,
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        # Daha eski SciPy sürümleri için geri dönüş.
        test = stats.mannwhitneyu(
            mel,
            nv,
            alternative="two-sided",
        )

    result["U"] = float(test.statistic)
    result["p_two"] = float(test.pvalue)
    return result


def holm_adjust(p_values, family_size):
    """Verilen test ailesi için Holm düzeltilmiş p değerlerini hesaplar."""
    result = {key: float("nan") for key in p_values}
    finite_items = [
        (key, float(value))
        for key, value in p_values.items()
        if value is not None and np.isfinite(value)
    ]

    if not finite_items:
        return result

    m = max(int(family_size), len(finite_items))
    ordered = sorted(finite_items, key=lambda item: item[1])
    running_max = 0.0

    for rank, (key, p_value) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, adjusted)
        result[key] = float(running_max)

    return result


def describe_group(group):
    if group == GROUP_NV:
        return "veri setinde iyi huylu nevüs (nv) olarak etiketlenen görseller"
    if group == GROUP_MEL:
        return "veri setinde kötü huylu melanom (mel) olarak etiketlenen görseller"
    return str(group)


def make_comment(item):
    """Ana A/B/C sonuçları için ayrıntılı, yönü açık grup yorumu üretir."""
    n_high = item["n_high"]
    n_low = item["n_low"]
    high_desc = describe_group(item["high_group"])
    low_desc = describe_group(item["low_group"])

    if n_high < 2 or n_low < 2:
        return (
            f"{high_desc} için geçerli n={n_high}, {low_desc} 
