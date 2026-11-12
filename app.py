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

_RESAMPLE = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)
_NEAREST = (
    Image.Resampling.NEAREST
    if hasattr(Image, "Resampling")
    else Image.NEAREST
)


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
# GÖRSEL İŞLEME VE ÖLÇÜM YARDIMCILARI
# ---------------------------------------------------------------------

def image_from_bytes(data):
    """Görsel baytlarını EXIF yönü düzeltilmiş RGB PIL görseline çevirir."""
    if data is None or len(data) == 0:
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
    elif isinstance(image, (bytes, bytearray, memoryview)):
        pil_image = image_from_bytes(image)
    else:
        raise TypeError("Desteklenmeyen görsel biçimi.")

    pil_image = pil_image.resize((IMG_N, IMG_N), _RESAMPLE)
    return np.asarray(pil_image, dtype=np.uint8)


def otsu_threshold(gray):
    """Gri tonlu görüntü için Otsu eşiğini hesaplar."""
    values = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    total = hist.sum()

    if total <= 0:
        return 127

    levels = np.arange(256, dtype=float)
    total_sum = float(np.dot(levels, hist))
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
        foreground_mean = (
            total_sum - background_sum
        ) / foreground_weight

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
    Merkezdeki uygun bileşeni; yoksa en büyük uygun bileşeni seçer.

    Bu otomatik maske yaklaşık bir bölütlemedir. Geçerli olması, lezyonu
    kusursuz biçimde çizdiği anlamına gelmez.
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

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    candidates = [
        label_id
        for label_id in range(1, count + 1)
        if min_area <= sizes[label_id] <= max_area
    ]

    if not candidates:
        return {
            "mask": np.zeros_like(gray, dtype=bool),
            "valid": False,
            "threshold": threshold,
            "area_fraction": 0.0,
        }

    center_label = int(labels[height // 2, width // 2])

    if (
        center_label > 0
        and min_area <= sizes[center_label] <= max_area
    ):
        selected_label = center_label
    else:
        selected_label = max(candidates, key=lambda label_id: sizes[label_id])

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
    """Varsayılan merkez geometriktir; isteğe bağlı maskenin ağırlık merkezi kullanılır."""
    height, width = rgb.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    if center_adjust:
        seg = segment_lesion(rgb)
        if seg["valid"] and np.any(seg["mask"]):
            ys, xs = np.where(seg["mask"])
            if len(xs):
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

    return binary[y1i, x1i], binary[y2i, x2i], valid


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
        return np.full((height, width), np.nan), coverage

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


def rotation_measure(
    binary,
    center,
    k=2,
    n_angles=180,
    step_px=4,
    with_heat=False,
):
    """Verilen ikili görüntü üzerinde dönel eşleşme, B(p) ve U hesaplar."""
    binary = np.asarray(binary)

    if binary.ndim != 2:
        raise ValueError("Dönel analiz için ikili görüntü iki boyutlu olmalıdır.")
    if not isinstance(k, (int, np.integer)) or int(k) < 2:
        raise ValueError("Dönel karşılaştırma mertebesi k en az 2 olmalıdır.")
    if not isinstance(n_angles, (int, np.integer)) or int(n_angles) < 1:
        raise ValueError("Açı örneği sayısı en az 1 olmalıdır.")
    if not isinstance(step_px, (int, np.integer)) or int(step_px) < 1:
        raise ValueError("Radyal örnekleme adımı en az 1 piksel olmalıdır.")

    binary = (binary > 0).astype(np.uint8)
    p = float(binary.mean())
    baseline_b = float(p * p + (1.0 - p) * (1.0 - p))

    radius_values = set(range(20, 81, int(step_px)))
    radius_values.update(SKOR_YARICAPLARI)
    radius_values = sorted(
        radius for radius in radius_values
        if 0 < radius <= 80
    )

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(n_angles),
        endpoint=False,
    )
    rotation_angle = 2.0 * np.pi / float(k)

    equal_count = 0
    compared_count = 0
    valid_pair_count = 0
    accepted_profiles = 0
    excluded_flat_profiles = 0
    possible_pair_count = len(angles) * len(radius_values)

    for theta in angles:
        first, second, valid = _sample_binary_pairs(
            binary,
            float(center[0]),
            float(center[1]),
            radius_values,
            theta,
            rotation_angle,
        )

        valid_count = int(np.count_nonzero(valid))
        valid_pair_count += valid_count

        coverage_at_angle = (
            valid_count / float(len(radius_values))
            if radius_values
            else 0.0
        )

        if coverage_at_angle < MIN_KAPSAMA or len(first) == 0:
            continue

        if DUZ_ALANLARI_DISLA:
            combined = np.concatenate([first, second])
            if combined.size and np.all(combined == combined[0]):
                excluded_flat_profiles += 1
                continue

        accepted_profiles += 1
        equal_count += int(np.count_nonzero(first == second))
        compared_count += int(len(first))

    score_s = (
        float(equal_count / compared_count)
        if compared_count
        else float("nan")
    )

    denominator = 1.0 - baseline_b
    if denominator > 1e-12 and np.isfinite(score_s):
        score_u = float((score_s - baseline_b) / denominator)
    else:
        score_u = float("nan")

    coverage = (
        float(valid_pair_count / possible_pair_count)
        if possible_pair_count
        else 0.0
    )

    if with_heat:
        heat, heat_coverage = local_u_map(
            binary,
            center,
            int(k),
            baseline_b,
        )
    else:
        heat = None
        heat_coverage = None

    return {
        "p": p,
        "B_chance": baseline_b,
        "S": score_s,
        "U": score_u,
        "k": int(k),
        "n_angles": int(n_angles),
        "step_px": int(step_px),
        "center": center,
        "coverage": coverage,
        "accepted_profiles": int(accepted_profiles),
        "excluded_flat_profiles": int(excluded_flat_profiles),
        "compared_pairs": int(compared_count),
        "heatmap": heat,
        "heat_coverage": heat_coverage,
    }


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
    tüm 200×200 görüntü gri tona çevrilir ve ortalama parlaklık eşiğiyle
    ikileştirilir. Bu ölçüm yalnızca lezyon maskesi üzerinde yapılmaz.
    """
    rgb = resize_rgb(image)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)

    threshold = float(gray.mean())
    binary = (gray >= threshold).astype(np.uint8)
    center = image_center(rgb, center_adjust=center_adjust)

    measures = rotation_measure(
        binary,
        center,
        k=k,
        n_angles=n_angles,
        step_px=step_px,
        with_heat=with_heat,
    )

    return {
        "rgb": rgb,
        "gray": gray,
        "binary": binary,
        "threshold": threshold,
        **measures,
    }


def mirror_mask_asymmetry(mask, axis):
    """
    Otsu maskesinin kendi ağırlık merkezi etrafındaki ayna örtüşmezliğini
    XOR / union oranıyla ölçer.

    vertical: sol-sağ yansıması, dikey eksen.
    horizontal: üst-alt yansıması, yatay eksen.
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


def mask_rotation_measure(mask, k=2, n_angles=180, step_px=4):
    """
    Otsu maskesi üzerinde ayrı, keşifsel dönel şekil ölçümü.
    Ana A ölçümünün yerine geçmez ve ana A/B/C test ailesine eklenmez.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2 or not np.any(mask):
        return {"S": float("nan"), "U": float("nan"), "coverage": 0.0}

    ys, xs = np.where(mask)
    center = (float(xs.mean()), float(ys.mean()))

    result = rotation_measure(
        mask.astype(np.uint8),
        center,
        k=k,
        n_angles=n_angles,
        step_px=step_px,
        with_heat=False,
    )

    return {
        "S": result["S"],
        "U": result["U"],
        "coverage": result["coverage"],
    }


def lesion_features(rgb, segmentation=None):
    """B, C ve keşifsel maske ayna-asimetrisi ölçümlerini hesaplar."""
    seg = segmentation or segment_lesion(rgb)

    result = {
        "segmentation_valid": bool(seg["valid"]),
        "area_fraction": float(seg["area_fraction"]),
        "B_edge": float("nan"),
        "C_colors": float("nan"),
        "color_names": "",
        "mirror_asym_lr": float("nan"),
        "mirror_asym_tb": float("nan"),
    }

    if not seg["valid"] or not np.any(seg["mask"]):
        return result

    mask = np.asarray(seg["mask"], dtype=bool)
    area = float(np.count_nonzero(mask))

    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = mask & (~eroded)
    perimeter = float(np.count_nonzero(boundary))

    result["B_edge"] = (
        (perimeter * perimeter) / (4.0 * math.pi * area)
        if area > 0
        else float("nan")
    )

    palette = np.asarray(
        [
            [25, 25, 25],
            [85, 50, 35],
            [160, 105, 65],
            [190, 45, 45],
            [95, 115, 145],
            [238, 235, 225],
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
        return result

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

    result.update(
        {
            "C_colors": float(len(present)),
            "color_names": ", ".join(present),
            "mirror_asym_lr": mirror_mask_asymmetry(mask, "vertical"),
            "mirror_asym_tb": mirror_mask_asymmetry(mask, "horizontal"),
        }
    )
    return result


def mask_overlay_array(rgb, mask):
    """Otsu maskesini görsel üzerine renkli katman ve sınır olarak bindirir."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)

    overlay = rgb.copy()
    if mask.ndim != 2 or mask.shape != rgb.shape[:2] or not np.any(mask):
        return overlay

    tint = np.asarray([235, 65, 85], dtype=float)
    overlay[mask] = np.uint8(
        0.58 * overlay[mask].astype(float) + 0.42 * tint
    )

    boundary = mask & (
        ~ndimage.binary_erosion(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
    )
    overlay[boundary] = np.asarray([255, 245, 60], dtype=np.uint8)
    return overlay


def dice_iou(prediction, reference):
    """İki ikili maskenin Dice ve IoU örtüşmesini hesaplar."""
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)

    if prediction.shape != reference.shape:
        raise ValueError("Tahmin ve referans maskeler aynı boyutta olmalıdır.")

    intersection = int(np.count_nonzero(prediction & reference))
    pred_count = int(np.count_nonzero(prediction))
    ref_count = int(np.count_nonzero(reference))
    union = int(np.count_nonzero(prediction | reference))

    if union == 0:
        return float("nan"), float("nan")

    dice = 2.0 * intersection / float(pred_count + ref_count)
    iou = intersection / float(union)
    return float(dice), float(iou)


def load_reference_mask(record):
    """Yüklenen siyah-beyaz elle çizilmiş maskeyi 200×200 ikili maske yapar."""
    data = record["uploaded"].getbuffer()

    try:
        with Image.open(io.BytesIO(data)) as source:
            gray = ImageOps.exif_transpose(source).convert("L")
            gray = gray.resize((IMG_N, IMG_N), _NEAREST)
            arr = np.asarray(gray, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Referans maskesi açılamadı: {exc}") from exc

    return arr > 127


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

    output = np.zeros((n, n, 3), dtype=np.uint8)
    output[:] = [245, 247, 244]

    foreground_level = np.clip(
        77.0 + 13.0 * (radius / max(float(radius.max()), 1e-6)) + texture,
        25,
        150,
    ).astype(np.uint8)

    output[..., 0] = np.where(inside, foreground_level, 245)
    output[..., 1] = np.where(inside, foreground_level - 6, 247)
    output[..., 2] = np.where(inside, foreground_level + 4, 244)

    return Image.fromarray(output)


def exact_synthetic_test_image(kind="symmetric", n=200):
    """Gürültüsüz, teknik gösterim amaçlı iki boyutlu sentetik şekil üretir."""
    yy, xx = np.mgrid[0:n, 0:n]
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0

    x = (xx - cx) / (n / 2.0)
    y = (yy - cy) / (n / 2.0)
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    if kind == "symmetric":
        boundary = 0.60 * (1.0 + 0.14 * np.cos(2.0 * theta))
    else:
        boundary = 0.60 * (
            1.0
            + 0.14 * np.cos(theta)
            + 0.07 * np.sin(3.0 * theta)
        )

    inside = radius <= boundary
    image = np.zeros((n, n, 3), dtype=np.uint8)
    image[:] = [245, 247, 244]
    image[inside] = [65, 50, 42]
    return Image.fromarray(image)


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
            if len(locations):
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


# ---------------------------------------------------------------------
# İSTATİSTİKSEL YARDIMCILAR
# ---------------------------------------------------------------------

def finite_values(values):
    """Sayısal ve sonlu değerleri güvenli biçimde numpy dizisine çevirir."""
    try:
        series = pd.Series(values)
    except Exception:
        series = pd.Series([], dtype=float)

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


def median_iqr(values):
    values = finite_values(values)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return float(median), float(q1), float(q3)


def median_iqr_text(values, digits=3):
    median, q1, q3 = median_iqr(values)
    if not np.isfinite(median):
        return "—"
    return (
        f"{fmt_number(median, digits)} "
        f"[{fmt_number(q1, digits)}, {fmt_number(q3, digits)}]"
    )


def welch_directional(expected_high, expected_low):
    """
    Önceden belirlenmiş yön için Welch t testi.
    Fark = beklenen yüksek grup ortalaması − beklenen düşük grup ortalaması.
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

    # Sıfır varyans durumunda p değerini yapay olarak sıfır yapmıyoruz.
    if se_squared <= 1e-24:
        return result

    standard_error = math.sqrt(se_squared)
    t_value = difference / standard_error

    denominator = 0.0
    denominator += (term_x * term_x) / (n1 - 1)
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
    pooled_variance = (
        ((n1 - 1) * var_x + (n2 - 1) * var_y) / pooled_denominator
    )
    pooled_sd = math.sqrt(max(pooled_variance, 0.0))
    cohen_d = (
        difference / pooled_sd
        if pooled_sd > 0
        else float("nan")
    )

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
    Beklenen yüksek grubun değerlerinin düşük gruptan yüksek olma
    sıralama olasılığı olan AUC'yi ve yaklaşık DeLong tipi GA'yı hesaplar.
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
    v10 = (
        y_left + 0.5 * (y_right - y_left)
    ) / float(n_low)

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
    z_value = float(
        stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    )

    result.update(
        {
            "auc_se": standard_error,
            "auc_ci_low": max(0.0, auc - z_value * standard_error),
            "auc_ci_high": min(1.0, auc + z_value * standard_error),
        }
    )
    return result


def mann_whitney_two_sided(sample_mel, sample_nv):
    """İki yönlü Mann–Whitney U testi; ilk örneklem mel grubudur."""
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
        test = stats.mannwhitneyu(
            mel,
            nv,
            alternative="two-sided",
        )

    result["U"] = float(test.statistic)
    result["p_two"] = float(test.pvalue)
    return result


def holm_adjust(p_values, family_size):
    """Bir test ailesi için Holm düzeltilmiş p değerlerini hesaplar."""
    adjusted = {key: float("nan") for key in p_values}
    finite_items = [
        (key, float(value))
        for key, value in p_values.items()
        if value is not None and np.isfinite(value)
    ]

    if not finite_items:
        return adjusted

    m = max(int(family_size), len(finite_items))
    ordered = sorted(finite_items, key=lambda item: item[1])
    running_max = 0.0

    for rank, (key, p_value) in enumerate(ordered):
        corrected = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, corrected)
        adjusted[key] = float(running_max)

    return adjusted


def describe_group(group):
    if group == GROUP_NV:
        return "veri setinde iyi huylu nevüs (nv) olarak etiketlenen görseller"
    if group == GROUP_MEL:
        return "veri setinde kötü huylu melanom (mel) olarak etiketlenen görseller"
    return str(group)


def make_comment(item):
    """Ana A/B/C sonuçları için grup adları açık, anlaşılır yorum üretir."""
    n_high = item["n_high"]
    n_low = item["n_low"]

    high_desc = describe_group(item["high_group"])
    low_desc = describe_group(item["low_group"])

    if n_high < 2 or n_low < 2:
        return (
            f"{high_desc} için geçerli n={n_high}, {low_desc} için "
            f"geçerli n={n_low}. Welch testi için iki grupta da en az "
            "iki geçerli ölçüm gerekir."
        )

    mean_high = item["mean_high"]
    mean_low = item["mean_low"]
    difference = item["difference"]

    base = (
        f"Bu yüklemede {high_desc} ortalaması {fmt_number(mean_high)}, "
        f"{low_desc} ortalaması {fmt_number(mean_low)} oldu. "
        f"Beklenen yönlü fark (yüksek grup − düşük grup) "
        f"{fmt_number(difference)}; olağan %95 güven aralığı "
        f"{fmt_ci(item['ci_low'], item['ci_high'])}."
    )

    if np.isfinite(item["p_one"]):
        test_text = (
            f"Beklenen yöndeki tek yönlü Welch p değeri "
            f"{fmt_p(item['p_one'])}; üç ana ölçüt için Bonferroni "
            f"eşiği α={ALFA_DUZELTILMIS:.4f}."
        )
    else:
        test_text = (
            "Bu örneklemde standart hata hesaplanamadığı için Welch "
            "p değeri üretilemedi."
        )

    if difference > 0:
        if (
            np.isfinite(item["p_one"])
            and item["p_one"] < ALFA_DUZELTILMIS
        ):
            conclusion = (
                "Gözlenen fark beklenen yönde ve yönlü p değeri "
                "Bonferroni eşiğinin altında. Bu, bu yüklemedeki etiketli "
                "gruplar için bir bulgudur; tek başına kanıt veya "
                "bireysel sınıflandırma değildir."
            )
        else:
            conclusion = (
                "Gözlenen fark beklenen yönde olsa da Bonferroni eşiğini "
                "geçmedi. Bu örneklemde beklenen yöndeki fark net biçimde "
                "gösterilemedi; bu, grupların aynı olduğunu kanıtlamaz."
            )
    elif difference < 0:
        conclusion = (
            "Gözlenen ortalama fark beklenen yönün tersinde. Bu sonuç "
            "ana hipotezin beklenen yönünü desteklemiyor; gruplar arasında "
            "hiç fark olmadığı sonucunu da vermez."
        )
    else:
        conclusion = (
            "Bu örneklemde grup ortalamaları eşit görünüyor. Bu, "
            "gruplar arasında kesinlikle fark olmadığı anlamına gelmez."
        )

    if np.isfinite(item["auc"]):
        auc_text = (
            f"Beklenen yönlü sıralama AUC'si {fmt_number(item['auc'], 3)} "
            f"(yaklaşık olağan %95 GA "
            f"{fmt_ci(item['auc_ci_low'], item['auc_ci_high'], 3)}). "
            "AUC p değerinin yerine geçmez; tanı doğruluğu olarak "
            "yorumlanmamalıdır."
        )
    else:
        auc_text = "AUC bu örneklemde hesaplanamadı."

    return " ".join([base, test_text, conclusion, auc_text])


def make_mirror_comment(item):
    """Keşifsel ayna-asimetrisi analizleri için anlaşılır yorum üretir."""
    if item["n_nv"] < 2 or item["n_mel"] < 2:
        return (
            f"Geçerli değer sayıları: nv={item['n_nv']}, mel={item['n_mel']}. "
            "İki grupta da en az ikişer değer olmadığından test yapılamıyor."
        )

    if not np.isfinite(item["auc"]):
        rank_text = "AUC hesaplanamadı."
    elif item["auc"] > 0.5:
        rank_text = (
            "Bu örneklemde daha yüksek ayna-asimetrisi değerleri, "
            "mel etiketli grupta daha sık sıralanma eğilimi gösteriyor."
        )
    elif item["auc"] < 0.5:
        rank_text = (
            "Bu örneklemde daha yüksek ayna-asimetrisi değerleri, "
            "nv etiketli grupta daha sık sıralanma eğilimi gösteriyor."
        )
    else:
        rank_text = (
            "Bu örneklemde AUC sıralama yönü 0,5 düzeyinde."
        )

    if np.isfinite(item["p_holm"]) and item["p_holm"] < 0.05:
        p_text = (
            "İki eksen için Holm düzeltmesinden sonraki p değeri "
            "0,05'in altında. Bu sonuç keşifseldir ve bağımsız veride "
            "ayrıca incelenmelidir."
        )
    elif np.isfinite(item["p_holm"]):
        p_text = (
            "Holm düzeltmesinden sonraki p değeri 0,05'in altında değil. "
            "Bu, iki grubun aynı olduğunu kanıtlamaz."
        )
    else:
        p_text = "Keşifsel p değeri hesaplanamadı."

    return (
        f"{item['axis_label']}: Otsu ile bulunan maske şeklinin kendi "
        f"merkezinden yansıtılmasıyla ölçülen örtüşmezlik değeridir; "
        f"yüksek değer daha az örtüşme anlamına gelir. "
        f"Medyanlar nv={fmt_number(item['median_nv'])}, "
        f"mel={fmt_number(item['median_mel'])}; medyan farkı "
        f"(mel−nv)={fmt_number(item['median_difference'])}. "
        f"{rank_text} AUC (mel>nv yönü) {fmt_number(item['auc'], 3)}, "
        f"yaklaşık olağan %95 GA "
        f"{fmt_ci(item['auc_ci_low'], item['auc_ci_high'], 3)}; "
        f"iki yönlü Mann–Whitney U={fmt_number(item['U'], 1)}, "
        f"ham p={fmt_p(item['p_raw'])}, iki eksen için Holm p="
        f"{fmt_p(item['p_holm'])}. {p_text}"
    )


# ---------------------------------------------------------------------
# TOPLU ANALİZ, DOSYA VE METADATA YARDIMCILARI
# ---------------------------------------------------------------------

def normalize_image_id(value):
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    return Path(text).stem.casefold()


def uploaded_records(files):
    """
    Yüklenen dosyaların kendisini tutar, tüm görsel baytlarını ayrıca
    kopyalayıp oturum sonuçlarında saklamaz.
    """
    records = []

    for uploaded in files or []:
        try:
            view = uploaded.getbuffer()
            if len(view) == 0:
                continue

            name = str(uploaded.name).replace("\\", "/")
            records.append(
                {
                    "name": Path(name).name,
                    "normalized_name": Path(name).name.casefold(),
                    "image_id": normalize_image_id(name),
                    "md5": hashlib.md5(view).hexdigest(),
                    "uploaded": uploaded,
                }
            )
        except Exception:
            continue

    return records


def duplicate_report(records_by_group):
    issues = []
    seen_md5 = {}
    seen_names = {}

    for group_name, records in records_by_group.items():
        for record in records:
            md5 = record["md5"]
            name = record["normalized_name"]

            if md5 in seen_md5:
                prior = seen_md5[md5]
                issues.append(
                    f"Aynı içerik iki kez yüklendi: "
                    f"{prior[1]} ve {record['name']}."
                )
            else:
                seen_md5[md5] = (group_name, record["name"])

            if name in seen_names:
                prior = seen_names[name]
                issues.append(
                    f"Dosya adı tekrarı: {prior[1]} ve {record['name']}."
                )
            else:
                seen_names[name] = (group_name, record["name"])

    return sorted(set(issues))


def duplicate_reference_names(records):
    counts = {}
    for record in records:
        image_id = record["image_id"]
        counts[image_id] = counts.get(image_id, 0) + 1

    return [
        image_id
        for image_id, count in counts.items()
        if image_id and count > 1
    ]


def reference_mask_map(records):
    result = {}
    duplicates = set()

    for record in records:
        key = record["image_id"]
        if not key:
            continue
        if key in result:
            duplicates.add(key)
        else:
            result[key] = record

    for key in duplicates:
        result.pop(key, None)

    return result


def check_metadata(metadata_bytes, nv_records, mel_records):
    result = {
        "provided": metadata_bytes is not None,
        "valid": False,
        "messages": [],
        "matched": 0,
        "image_info": {},
        "unique_nv_lesions": 0,
        "unique_mel_lesions": 0,
    }

    if metadata_bytes is None:
        result["messages"].append(
            "Metadata CSV yüklenmedi; image_id, dx ve lesion_id "
            "eşleşmeleri kontrol edilemedi."
        )
        return result

    try:
        metadata = pd.read_csv(
            io.BytesIO(metadata_bytes),
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        result["messages"].append(f"Metadata CSV okunamadı: {exc}")
        return result

    column_lookup = {
        str(column).strip().casefold(): column
        for column in metadata.columns
    }
    required = ["image_id", "lesion_id", "dx"]
    missing_columns = [
        column for column in required
        if column not in column_lookup
    ]

    if missing_columns:
        result["messages"].append(
            "Metadata CSV içinde gereken sütunlar bulunamadı: "
            + ", ".join(missing_columns)
            + ". Gerekli sütunlar: image_id, lesion_id, dx."
        )
        return result

    image_col = column_lookup["image_id"]
    lesion_col = column_lookup["lesion_id"]
    dx_col = column_lookup["dx"]

    metadata = metadata.copy()
    metadata["_image"] = metadata[image_col].map(normalize_image_id)
    metadata["_lesion"] = (
        metadata[lesion_col].astype(str).str.strip().str.casefold()
    )
    metadata["_dx"] = (
        metadata[dx_col].astype(str).str.strip().str.casefold()
    )

    duplicate_ids = metadata["_image"].duplicated(keep=False)
    if duplicate_ids.any():
        result["messages"].append(
            "Metadata içinde tekrarlanan image_id bulundu; eşleşme belirsiz."
        )
        return result

    lookup = {
        row["_image"]: {
            "lesion_id": row["_lesion"],
            "dx": row["_dx"],
        }
        for _, row in metadata.iterrows()
        if row["_image"]
    }

    missing_files = []
    mismatched_labels = []
    missing_lesions = []
    selected = []

    for group_name, records, expected_dx in [
        (GROUP_NV, nv_records, "nv"),
        (GROUP_MEL, mel_records, "mel"),
    ]:
        for record in records:
            image_id = record["image_id"]
            row = lookup.get(image_id)

            if row is None:
                missing_files.append(record["name"])
                continue

            result["matched"] += 1
            actual_dx = row["dx"]
            lesion_id = row["lesion_id"]

            result["image_info"][image_id] = {
                "lesion_id": lesion_id,
                "dx": actual_dx,
            }

            if actual_dx != expected_dx:
                mismatched_labels.append(
                    f"{record['name']} için dx='{actual_dx}', "
                    f"beklenen '{expected_dx}'."
                )

            if not lesion_id:
                missing_lesions.append(record["name"])
            else:
                selected.append(
                    {
                        "group": group_name,
                        "lesion_id": lesion_id,
                    }
                )

    if missing_files:
        result["messages"].append(
            f"Metadata ile eşleşmeyen görsel sayısı: {len(missing_files)}."
        )

    if mismatched_labels:
        result["messages"].append(
            f"Grup etiketiyle uyuşmayan dx sayısı: {len(mismatched_labels)}."
        )

    if missing_lesions:
        result["messages"].append(
            f"lesion_id değeri boş olan görsel sayısı: {len(missing_lesions)}."
        )

    duplicate_groups = []
    lesion_sets = {}

    for group_name in [GROUP_NV, GROUP_MEL]:
        ids = [
            item["lesion_id"]
            for item in selected
            if item["group"] == group_name
        ]
        lesion_sets[group_name] = set(ids)

        if len(ids) != len(set(ids)):
            duplicate_groups.append(group_name)

    if duplicate_groups:
        result["messages"].append(
            "Aynı grupta birden fazla görsel aynı lesion_id değerine sahip."
        )

    cross_group = (
        lesion_sets.get(GROUP_NV, set())
        & lesion_sets.get(GROUP_MEL, set())
    )
    if cross_group:
        result["messages"].append(
            "İki grup arasında ortak lesion_id bulundu."
        )

    result["unique_nv_lesions"] = len(lesion_sets.get(GROUP_NV, set()))
    result["unique_mel_lesions"] = len(lesion_sets.get(GROUP_MEL, set()))

    valid = (
        bool(nv_records)
        and bool(mel_records)
        and not missing_files
        and not mismatched_labels
        and not missing_lesions
        and not duplicate_groups
        and not cross_group
    )
    result["valid"] = bool(valid)

    if valid:
        result["messages"].append(
            "Yüklenen görsellerin image_id/dx eşleşmesi ve gruplar içindeki "
            "lesion_id tekilliği kontrol edildi."
        )

    result["messages"].append(
        "Metadata, örneklerin tohum 42 ile seçildiğini tek başına göstermez; "
        "örnek seçim kaydı ayrıca saklanmalıdır."
    )
    return result


def batch_signature(
    nv_records,
    mel_records,
    metadata_bytes,
    reference_records=None,
):
    parts = [SURUM]

    for group_name, records in [
        ("nv", nv_records),
        ("mel", mel_records),
        ("reference_mask", reference_records or []),
    ]:
        for record in records:
            parts.append(
                f"{group_name}:{record['normalized_name']}:{record['md5']}"
            )

    if metadata_bytes is not None:
        parts.append(
            "metadata:" + hashlib.md5(metadata_bytes).hexdigest()
        )

    return hashlib.sha256(
        "|".join(sorted(parts)).encode("utf-8")
    ).hexdigest()


def analyze_batch_group(
    records,
    group_name,
    image_info=None,
    mask_lookup=None,
):
    rows = []
    errors = []
    image_info = image_info or {}
    mask_lookup = mask_lookup or {}

    for record in records:
        try:
            image = image_from_bytes(record["uploaded"].getbuffer())
            analysis = gorsel_analiz(
                image,
                k=2,
                n_angles=180,
                step_px=4,
                center_adjust=False,
                with_heat=False,
            )

            seg = segment_lesion(analysis["rgb"])
            features = lesion_features(analysis["rgb"], seg)

            mask_rotation = (
                mask_rotation_measure(
                    seg["mask"],
                    k=2,
                    n_angles=180,
                    step_px=4,
                )
                if seg["valid"]
                else {"S": float("nan"), "U": float("nan")}
            )

            lesion_id = image_info.get(
                record["image_id"],
                {},
            ).get("lesion_id", "")

            reference_record = mask_lookup.get(record["image_id"])
            reference_dice = float("nan")
            reference_iou = float("nan")
            reference_filename = ""

            if reference_record is not None:
                reference_mask = load_reference_mask(reference_record)
                reference_dice, reference_iou = dice_iou(
                    seg["mask"],
                    reference_mask,
                )
                reference_filename = reference_record["name"]

            rows.append(
                {
                    "grup": group_name,
                    "dosya": record["name"],
                    "lesion_id": lesion_id,
                    "A_U": analysis["U"],
                    "A_S": analysis["S"],
                    "B_p": analysis["B_chance"],
                    "p_acik_piksel": analysis["p"],
                    "kapsama": analysis["coverage"],
                    "B_edge": features["B_edge"],
                    "C_colors": features["C_colors"],
                    "renk_siniflari": features["color_names"],
                    "A_mask_U": mask_rotation["U"],
                    "mirror_asym_lr": features["mirror_asym_lr"],
                    "mirror_asym_tb": features["mirror_asym_tb"],
                    "segmentasyon_gecerli": features["segmentation_valid"],
                    "lezyon_alan_orani": features["area_fraction"],
                    "referans_maskesi": reference_filename,
                    "referans_Dice": reference_dice,
                    "referans_IoU": reference_iou,
                }
            )
        except Exception as exc:
            errors.append(f"{record['name']}: {exc}")

    return rows, errors


def run_batch(
    nv_records,
    mel_records,
    image_info=None,
    mask_lookup=None,
):
    nv_rows, nv_errors = analyze_batch_group(
        nv_records,
        GROUP_NV,
        image_info=image_info,
        mask_lookup=mask_lookup,
    )
    mel_rows, mel_errors = analyze_batch_group(
        mel_records,
        GROUP_MEL,
        image_info=image_info,
        mask_lookup=mask_lookup,
    )

    return {
        "data": pd.DataFrame(nv_rows + mel_rows),
        "errors": nv_errors + mel_errors,
        "nv_processed": len(nv_rows),
        "mel_processed": len(mel_rows),
    }


# ---------------------------------------------------------------------
# TOPLU İSTATİSTİK VE KALİTE ÖZETLERİ
# ---------------------------------------------------------------------

def build_batch_statistics(data):
    metric_specs = [
        {
            "metric": "A — Dönel simetri ölçümü (U)",
            "column": "A_U",
            "high_group": GROUP_NV,
            "low_group": GROUP_MEL,
            "direction": "İyi huylu nevüs (nv) > kötü huylu melanom (mel)",
            "difference_label": "nv − mel",
        },
        {
            "metric": "B — Kenar düzensizliği indeksi",
            "column": "B_edge",
            "high_group": GROUP_MEL,
            "low_group": GROUP_NV,
            "direction": "Kötü huylu melanom (mel) > iyi huylu nevüs (nv)",
            "difference_label": "mel − nv",
        },
        {
            "metric": "C — Renk sınıfı sayısı",
            "column": "C_colors",
            "high_group": GROUP_MEL,
            "low_group": GROUP_NV,
            "direction": "Kötü huylu melanom (mel) > iyi huylu nevüs (nv)",
            "difference_label": "mel − nv",
        },
    ]

    results = []

    for spec in metric_specs:
        high = finite_values(
            data.loc[
                data["grup"] == spec["high_group"],
                spec["column"],
            ]
        )
        low = finite_values(
            data.loc[
                data["grup"] == spec["low_group"],
                spec["column"],
            ]
        )

        test = welch_directional(high, low)
        auc = auc_delong_ci(high, low)

        nv_values = finite_values(
            data.loc[data["grup"] == GROUP_NV, spec["column"]]
        )
        mel_values = finite_values(
            data.loc[data["grup"] == GROUP_MEL, spec["column"]]
        )

        nv_median, nv_q1, nv_q3 = median_iqr(nv_values)
        mel_median, mel_q1, mel_q3 = median_iqr(mel_values)

        item = {
            **spec,
            **test,
            **auc,
            "nv_mean": (
                float(np.mean(nv_values))
                if len(nv_values)
                else float("nan")
            ),
            "mel_mean": (
                float(np.mean(mel_values))
                if len(mel_values)
                else float("nan")
            ),
            "nv_median": nv_median,
            "nv_q1": nv_q1,
            "nv_q3": nv_q3,
            "mel_median": mel_median,
            "mel_q1": mel_q1,
            "mel_q3": mel_q3,
        }

        item["comment"] = make_comment(item)
        item["significant"] = (
            item["n_high"] >= 2
            and item["n_low"] >= 2
            and np.isfinite(item["p_one"])
            and item["difference"] > 0
            and item["p_one"] < ALFA_DUZELTILMIS
        )
        results.append(item)

    return results


def results_table(results):
    rows = []

    for item in results:
        rows.append(
            {
                "Ölçüt": item["metric"],
                "Beklenen yön": item["direction"],
                "İyi huylu nevüs (nv) ortalaması": fmt_number(item["nv_mean"]),
                "İyi huylu nevüs (nv) medyan [Q1, Q3]": (
                    f"{fmt_number(item['nv_median'])} "
                    f"[{fmt_number(item['nv_q1'])}, {fmt_number(item['nv_q3'])}]"
                    if np.isfinite(item["nv_median"])
                    else "—"
                ),
                "Kötü huylu melanom (mel) ortalaması": fmt_number(item["mel_mean"]),
                "Kötü huylu melanom (mel) medyan [Q1, Q3]": (
                    f"{fmt_number(item['mel_median'])} "
                    f"[{fmt_number(item['mel_q1'])}, {fmt_number(item['mel_q3'])}]"
                    if np.isfinite(item["mel_median"])
                    else "—"
                ),
                f"Beklenen yönlü fark ({item['difference_label']})": (
                    fmt_number(item["difference"])
                ),
                "Welch t": fmt_number(item["t"], 3),
                "Tek yönlü p": fmt_p(item["p_one"]),
                "İki yönlü p": fmt_p(item["p_two"]),
                "Cohen d": fmt_number(item["cohen_d"], 3),
                "Fark için olağan %95 GA": fmt_ci(
                    item["ci_low"],
                    item["ci_high"],
                ),
                "AUC (beklenen yön)": fmt_number(item["auc"], 3),
                "AUC olağan %95 GA": fmt_ci(
                    item["auc_ci_low"],
                    item["auc_ci_high"],
                    3,
                ),
                "Bonferroni eşiği": f"{ALFA_DUZELTILMIS:.4f}",
                "Geçerli n (yüksek/düşük)": (
                    f"{item['n_high']}/{item['n_low']}"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_mirror_statistics(data):
    """İki keşifsel ayna ölçümünü ana A/B/C analizinden ayrı hesaplar."""
    specs = [
        {
            "code": "lr",
            "axis_label": "Sol-sağ yansıması (dikey eksen)",
            "column": "mirror_asym_lr",
        },
        {
            "code": "tb",
            "axis_label": "Üst-alt yansıması (yatay eksen)",
            "column": "mirror_asym_tb",
        },
    ]

    results = []

    for spec in specs:
        nv = finite_values(
            data.loc[data["grup"] == GROUP_NV, spec["column"]]
        )
        mel = finite_values(
            data.loc[data["grup"] == GROUP_MEL, spec["column"]]
        )

        mw = mann_whitney_two_sided(mel, nv)
        auc = auc_delong_ci(mel, nv)

        median_nv = float(np.median(nv)) if len(nv) else float("nan")
        median_mel = float(np.median(mel)) if len(mel) else float("nan")

        results.append(
            {
                **spec,
                **mw,
                **auc,
                "median_nv": median_nv,
                "median_mel": median_mel,
                "median_difference": (
                    median_mel - median_nv
                    if np.isfinite(median_nv) and np.isfinite(median_mel)
                    else float("nan")
                ),
                "p_raw": mw["p_two"],
            }
        )

    adjusted = holm_adjust(
        {item["code"]: item["p_raw"] for item in results},
        family_size=2,
    )

    for item in results:
        item["p_holm"] = adjusted.get(item["code"], float("nan"))
        item["comment"] = make_mirror_comment(item)

    return results


def mirror_results_table(results):
    rows = []

    for item in results:
        rows.append(
            {
                "Keşifsel ölçüt": item["axis_label"],
                "İyi huylu nevüs (nv) medyanı": fmt_number(item["median_nv"]),
                "Kötü huylu melanom (mel) medyanı": fmt_number(item["median_mel"]),
                "Medyan farkı (mel − nv)": fmt_number(item["median_difference"]),
                "Mann–Whitney U (mel ilk grup)": fmt_number(item["U"], 1),
                "İki yönlü ham p": fmt_p(item["p_raw"]),
                "İki eksen için Holm p": fmt_p(item["p_holm"]),
                "AUC (mel > nv)": fmt_number(item["auc"], 3),
                "AUC olağan %95 GA": fmt_ci(
                    item["auc_ci_low"],
                    item["auc_ci_high"],
                    3,
                ),
                "Geçerli n (mel/nv)": (
                    f"{item['n_mel']}/{item['n_nv']}"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_mask_rotation_summary(data):
    """Keşifsel maske dönel U değerlerini yalnızca betimsel olarak özetler."""
    rows = []

    for group_name in [GROUP_NV, GROUP_MEL]:
        values = finite_values(
            data.loc[data["grup"] == group_name, "A_mask_U"]
        )
        median, q1, q3 = median_iqr(values)

        rows.append(
            {
                "Grup": group_name,
                "Geçerli n": len(values),
                "Medyan [Q1, Q3]": (
                    f"{fmt_number(median)} "
                    f"[{fmt_number(q1)}, {fmt_number(q3)}]"
                    if np.isfinite(median)
                    else "—"
                ),
            }
        )

    return pd.DataFrame(rows)


def batch_quality_summary(data):
    """Segmentasyon, alan oranı, açık piksel oranı ve elle maske ölçülerini özetler."""
    rows = []

    for group_name in [GROUP_NV, GROUP_MEL]:
        group_data = data.loc[data["grup"] == group_name]
        total = len(group_data)

        if "segmentasyon_gecerli" in group_data.columns:
            valid = (
                group_data["segmentasyon_gecerli"]
                .fillna(False)
                .astype(bool)
            )
        else:
            valid = pd.Series(False, index=group_data.index)

        valid_count = int(valid.sum())
        valid_percent = (
            100.0 * valid_count / total
            if total
            else float("nan")
        )

        area_values = finite_values(
            group_data.loc[valid, "lezyon_alan_orani"]
            if "lezyon_alan_orani" in group_data.columns
            else []
        )
        p_values = finite_values(
            group_data["p_acik_piksel"]
            if "p_acik_piksel" in group_data.columns
            else []
        )
        dice_values = finite_values(
            group_data["referans_Dice"]
            if "referans_Dice" in group_data.columns
            else []
        )
        iou_values = finite_values(
            group_data["referans_IoU"]
            if "referans_IoU" in group_data.columns
            else []
        )

        rows.append(
            {
                "Grup": group_name,
                "İşlenen görsel": total,
                "Geçerli Otsu segmentasyonu": valid_count,
                "Geçerli segmentasyon (%)": (
                    fmt_number(valid_percent, 1)
                    if np.isfinite(valid_percent)
                    else "—"
                ),
                "Otsu lezyon alan oranı, medyan [Q1, Q3]": (
                    median_iqr_text(area_values)
                ),
                "A için açık piksel oranı p, medyan [Q1, Q3]": (
                    median_iqr_text(p_values)
                ),
                "Elle maske eşleşen n": len(dice_values),
                "Referans maskeye Dice, ortalama": (
                    fmt_number(np.mean(dice_values), 3)
                    if len(dice_values)
                    else "—"
                ),
                "Referans maskeye IoU, ortalama": (
                    fmt_number(np.mean(iou_values), 3)
                    if len(iou_values)
                    else "—"
                ),
            }
        )

    return pd.DataFrame(rows)


def make_mask_review_sample(data, per_group=25, seed=42):
    """
    Elle maskeyle kalite kontrolü için grup başına en fazla 25 görsel
    seçer. Bu seçim ana analiz örneklemini değiştirmez.
    """
    pieces = []

    for group_name in [GROUP_NV, GROUP_MEL]:
        group_data = data.loc[data["grup"] == group_name].copy()

        if group_data.empty:
            continue

        if "lesion_id" in group_data.columns:
            ids = group_data["lesion_id"].fillna("").astype(str).str.strip()
            with_id = group_data.loc[ids.ne("")].copy()

            if not with_id.empty:
                group_data = with_id.drop_duplicates(
                    subset=["lesion_id"],
                    keep="first",
                )

        take = min(int(per_group), len(group_data))
        if take <= 0:
            continue

        sample = group_data.sample(n=take, random_state=seed)
        keep_columns = [
            column
            for column in ["grup", "dosya", "lesion_id"]
            if column in sample.columns
        ]
        pieces.append(sample[keep_columns])

    if not pieces:
        return pd.DataFrame(columns=["grup", "dosya", "lesion_id"])

    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------
# Matplotlib sürümünden bağımsız grup grafikleri
# ---------------------------------------------------------------------

def draw_manual_box(ax, values, position, color):
    """Matplotlib boxplot API'sine bağlı olmayan kutu grafiği çizer."""
    values = finite_values(values)
    if len(values) == 0:
        return

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inliers = values[
        (values >= lower_fence)
        & (values <= upper_fence)
    ]

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

    outliers = values[
        (values < lower_fence)
        | (values > upper_fence)
    ]

    if len(outliers):
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


def batch_boxplot(data, specs, figsize):
    fig, axes = plt.subplots(1, len(specs), figsize=figsize)
    axes = np.atleast_1d(axes)

    groups = [
        (GROUP_NV, GROUP_NV_CHART, "#55a9a2"),
        (GROUP_MEL, GROUP_MEL_CHART, "#8b72bd"),
    ]

    for ax, (title, column) in zip(axes, specs):
        values_by_group = []

        for group_name, label, color in groups:
            if column in data.columns:
                source = data.loc[data["grup"] == group_name, column]
            else:
                source = []

            values_by_group.append(
                (label, color, finite_values(source))
            )

        if any(len(values) == 0 for _, _, values in values_by_group):
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

        for position, (_, color, values) in enumerate(
            values_by_group,
            start=1,
        ):
            draw_manual_box(ax, values, position, color)

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
    return batch_boxplot(
        data,
        [
            ("A — Dönel simetri U", "A_U"),
            ("B — Kenar düzensizliği", "B_edge"),
            ("C — Renk sınıfı sayısı", "C_colors"),
        ],
        figsize=(15, 4.8),
    )


def fig_mirror_boxplots(data):
    return batch_boxplot(
        data,
        [
            ("Keşifsel sol-sağ ayna asimetrisi", "mirror_asym_lr"),
            ("Keşifsel üst-alt ayna asimetrisi", "mirror_asym_tb"),
        ],
        figsize=(10.5, 4.8),
    )


# ---------------------------------------------------------------------
# TEKLİ GÖRSEL YORUMLARI
# ---------------------------------------------------------------------

def describe_u(value):
    """U değerini tanı veya sınıflandırma vermeden sade biçimde açıklar."""
    if value is None or not np.isfinite(value):
        return (
            "U hesaplanamadı. Geçerli örnek çiftleri bulunmamış veya "
            "normalizasyon için yeterli değişkenlik oluşmamış olabilir."
        )

    if value > 1e-12:
        return (
            "U pozitiftir: bu ayarlarda ölçülen dönel eşleşme, "
            "görselin kendi B(p) tabanının üzerindedir."
        )
    if value < -1e-12:
        return (
            "U negatiftir: bu ayarlarda ölçülen dönel eşleşme, "
            "görselin kendi B(p) tabanının altındadır."
        )
    return (
        "U sıfıra çok yakındır: ölçülen eşleşme, bu hesaplamadaki "
        "B(p) tabanına yakındır."
    )


def render_visual_panel(title, analysis):
    """Tekli/ikili analizde görsel, maske, ısı haritası ve açıklamaları gösterir."""
    rgb = analysis["rgb"]
    seg = segment_lesion(rgb)
    features = lesion_features(rgb, seg)

    mask_rotation = (
        mask_rotation_measure(
            seg["mask"],
            k=analysis["k"],
            n_angles=analysis["n_angles"],
            step_px=analysis["step_px"],
        )
        if seg["valid"]
        else {"S": float("nan"), "U": float("nan"), "coverage": 0.0}
    )

    st.write(f"**{title}**")

    image_col, detail_col = st.columns(2)

    with image_col:
        st.image(
            rgb,
            caption="Analizde kullanılan 200×200 RGB görüntü",
            use_container_width=True,
        )
        st.image(
            mask_overlay_array(rgb, seg["mask"]),
            caption=(
                "Otsu ile seçilen alan: renkli bölge; sarı çizgi: yaklaşık sınır"
            ),
            use_container_width=True,
        )

    with detail_col:
        st.image(
            analysis["binary"] * 255,
            caption="A ölçümü için ortalama parlaklık eşiğiyle ikileştirme",
            use_container_width=True,
        )
        fig = fig_heat_overlay(
            rgb,
            analysis["heatmap"],
            "Yerel U ısı haritası",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("A — Tüm görüntüde U", fmt_number(analysis["U"]))
    m2.metric("B — Kenar düzensizliği", fmt_number(features["B_edge"]))
    m3.metric("C — Renk sınıfı sayısı", fmt_number(features["C_colors"], 0))
    m4.metric("Keşifsel maske U", fmt_number(mask_rotation["U"]))

    st.info(describe_u(analysis["U"]))

    st.caption(
        f"A ayarları: k={analysis['k']} ({K_SECENEKLERI[analysis['k']]}), "
        f"{analysis['n_angles']} açı, {analysis['step_px']} px adım. "
        f"S={fmt_number(analysis['S'])}, B(p)={fmt_number(analysis['B_chance'])}, "
        f"açık piksel oranı p={fmt_number(analysis['p'])}, "
        f"geçerli örnek çiftlerinin kapsaması={analysis['coverage']:.1%}."
    )

    if seg["valid"]:
        st.caption(
            f"Otsu maskesi yaklaşık alanın %{100 * seg['area_fraction']:.1f}'ini "
            f"kapsıyor. Segmentasyon geçerli sayılmıştır; bu, sınırın doğru "
            "olduğunu tek başına kanıtlamaz."
        )
    else:
        st.warning(
            "Otsu yöntemi bu görselde geçerli lezyon maskesi bulamadı. "
            "B, C ve maske temelli keşifsel değerler güvenilir olmayabilir."
        )

    if analysis["heatmap"] is None or not np.isfinite(analysis["heatmap"]).any():
        heat_text = "Isı haritasında gösterilecek geçerli yerel değer yok."
    else:
        heat_text = (
            "Isı haritasında kırmızı tonlar yerel U'nun B(p) tabanının "
            "üzerinde; mavi tonlar altında olduğu bölgeleri gösterir. "
            "Boş bölgelerde geçerli yerel değer hesaplanmamıştır."
        )

    st.caption(heat_text)
    st.caption(
        "A ölçümü tüm 200×200 görüntüden hesaplanır; arka plan, kadraj ve "
        "aydınlatmadan etkilenebilir. Keşifsel maske U ise Otsu maskesinin "
        "şeklinden hesaplanır ve ana A ölçümünün yerine geçmez."
    )

    detail = pd.DataFrame(
        [
            {"Ölçüm": "A — Eşleşme oranı S", "Değer": analysis["S"]},
            {"Ölçüm": "A — Görüntüye ait B(p)", "Değer": analysis["B_chance"]},
            {"Ölçüm": "A — Açık piksel oranı p", "Değer": analysis["p"]},
            {"Ölçüm": "A — Normalize dönel U", "Değer": analysis["U"]},
            {"Ölçüm": "B — Kenar düzensizliği", "Değer": features["B_edge"]},
            {"Ölçüm": "C — Renk sınıfı sayısı", "Değer": features["C_colors"]},
            {"Ölçüm": "Keşifsel — Maske dönel U", "Değer": mask_rotation["U"]},
            {
                "Ölçüm": "Keşifsel — Sol-sağ maske asimetrisi",
                "Değer": features["mirror_asym_lr"],
            },
            {
                "Ölçüm": "Keşifsel — Üst-alt maske asimetrisi",
                "Değer": features["mirror_asym_tb"],
            },
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    return {
        "segmentation": seg,
        "features": features,
        "mask_rotation": mask_rotation,
    }


def compare_single_results(result_a, panel_a, result_b, panel_b):
    """İki tekil görselin ölçüm farkını ve sade açıklamasını gösterir."""
    delta_u = result_b["U"] - result_a["U"]

    st.markdown("#### İki görselin ölçüm karşılaştırması")

    if np.isfinite(delta_u):
        if delta_u > 0:
            summary = (
                f"Aynı ayarlarda Görsel 2'nin U değeri Görsel 1'den "
                f"{fmt_number(delta_u)} daha yüksektir."
            )
        elif delta_u < 0:
            summary = (
                f"Aynı ayarlarda Görsel 1'in U değeri Görsel 2'den "
                f"{fmt_number(abs(delta_u))} daha yüksektir."
            )
        else:
            summary = "İki görselin U ölçümleri eşittir."
    else:
        summary = "U ölçümü hesaplanamadığı için doğrudan karşılaştırma yapılamıyor."

    st.info(
        summary
        + " U her görselin kendi B(p) tabanına göre hesaplanır; bu karşılaştırma "
        "istatistiksel grup testi veya sınıflandırma değildir."
    )

    rows = []

    comparison_specs = [
        ("A — Tüm görüntüde U", result_a["U"], result_b["U"]),
        ("A — Eşleşme oranı S", result_a["S"], result_b["S"]),
        ("A — Görüntüye ait B(p)", result_a["B_chance"], result_b["B_chance"]),
        (
            "B — Kenar düzensizliği",
            panel_a["features"]["B_edge"],
            panel_b["features"]["B_edge"],
        ),
        (
            "C — Renk sınıfı sayısı",
            panel_a["features"]["C_colors"],
            panel_b["features"]["C_colors"],
        ),
        (
            "Keşifsel — Maske dönel U",
            panel_a["mask_rotation"]["U"],
            panel_b["mask_rotation"]["U"],
        ),
        (
            "Keşifsel — Sol-sağ maske asimetrisi",
            panel_a["features"]["mirror_asym_lr"],
            panel_b["features"]["mirror_asym_lr"],
        ),
        (
            "Keşifsel — Üst-alt maske asimetrisi",
            panel_a["features"]["mirror_asym_tb"],
            panel_b["features"]["mirror_asym_tb"],
        ),
    ]

    for name, value_a, value_b in comparison_specs:
        try:
            difference = float(value_b) - float(value_a)
        except (TypeError, ValueError):
            difference = float("nan")

        rows.append(
            {
                "Ölçüm": name,
                "Görsel 1": value_a,
                "Görsel 2": value_b,
                "Fark (Görsel 2 − Görsel 1)": difference,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------
# İSTATİSTİK GRAFİKLERİ
# ---------------------------------------------------------------------

def fig_industry_overlay(rgb, heat, title):
    return fig_heat_overlay(rgb, heat, title, show_focus=True)


# ---------------------------------------------------------------------
# SOL AYAR ÇUBUĞU
# ---------------------------------------------------------------------

st.sidebar.markdown("## ✳️ YSÜM")
st.sidebar.caption("Görüntülerde dönel simetriyi inceleyen proje prototipi")

k = st.sidebar.selectbox(
    "Dönel karşılaştırma mertebesi",
    options=list(K_DEGERLERI),
    index=0,
    format_func=lambda value: K_SECENEKLERI[value],
)

hassasiyet = st.sidebar.selectbox(
    "Örnekleme hassasiyeti",
    options=list(HASSASIYETLER.keys()),
    index=1,
)

n_angles, step_px = HASSASIYETLER[hassasiyet]

center_adjust = st.sidebar.checkbox(
    "Merkezi koyu bileşene göre ayarla",
    value=False,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Önemli ayrım**

    `k=2`, görüntüyü 180° döndürerek karşılaştırır.
    Ayna/eksen yansıması ölçümü değildir.
    """
)
st.sidebar.caption(
    f"{SURUM} · {n_angles} açı örneği · {step_px} px radyal adım"
)


# ---------------------------------------------------------------------
# SAYFA BAŞLIĞI VE SEKME YAPISI
# ---------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Görüntü ölçümü · Proje prototipi · {SURUM}</div>
        <h1>YSÜM — Yerel Simetri Üstünlük Modeli</h1>
        <p>
            Görüntülerin dönel benzerliğini, yerel U haritasını ve ek görüntü
            ölçütlerini inceleyen araştırma amaçlı görsel analiz prototipi.
            Klinik tanı veya endüstriyel uygunluk kararı vermez.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "📐 Kuramsal Model",
        "🧪 Kalibrasyon & Doğrulama",
        "🖼️ Görsel Analiz & Isı Haritası",
        "🏭 Endüstriyel & Geometrik İnceleme",
        "📊 HAM10000 Grup Karşılaştırması",
        "🧭 Model Mimarisi & Metodoloji",
    ]
)


# ---------------------------------------------------------------------
# 1. KURAMSAL MODEL
# ---------------------------------------------------------------------

with tabs[0]:
    st.markdown("### Altın oranla kuramsal bağlantı")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    p_ref = 1.0 / phi
    b_ref = p_ref**2 + (1.0 - p_ref) ** 2

    c1, c2, c3 = st.columns(3)
    c1.metric("Altın oran φ", f"{phi:.6f}")
    c2.metric("Kuramsal açık oran 1/φ", f"{p_ref:.6f}")
    c3.metric("B(1/φ) referansı", f"{b_ref:.6f}")

    st.markdown(
        """
        <div class="section-card">
        <b>Bu bağlantı ne anlatıyor?</b><br>
        Modelde Penrose/altın oran bağlantısı, <code>p = 1/φ</code> seçildiğinde
        <code>B(p) = p² + (1−p)² ≈ 0,528</code> elde edilmesine dayanır.
        Bu değer kuramsal bir referanstır; gerçek görüntüler için evrensel
        simetri eşiği olduğunu kanıtlamaz. Görsel analizinde her görüntünün
        kendi ikili görüntüsünden hesaplanan <code>B(p)</code> kullanılır.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.pyplot(fig_penrose_baseline(), use_container_width=True)

    with col2:
        st.markdown("#### Modelde kullanılan dönüşüm")
        st.latex(r"B(p)=p^2+(1-p)^2")
        st.latex(r"U_k=\frac{S_k-B(p)}{1-B(p)}")
        st.markdown(
            """
            - `Sₖ`: k katlı dönel eşleşme oranı.
            - `B(p)`: görüntünün kendi açık piksel oranından hesaplanan şans düzeyi.
            - `Uₖ`: eşleşmeyi bu şans düzeyine göre normalize eden ölçüm.
            """
        )
        st.caption(
            f"Yorum kolaylığı için kullanılan |U|≤{YORUM_ESIGI:.2f} bandı "
            "istatistiksel anlamlılık eşiği değildir."
        )

    st.markdown("#### Özyinelemeli oran örneği")
    rows = []
    K_n, I_n = 1, 0

    for n in range(8):
        total = K_n + I_n
        rows.append(
            {
                "n": n,
                "Kₙ": K_n,
                "Iₙ": I_n,
                "Kₙ / (Kₙ + Iₙ)": K_n / total if total else np.nan,
            }
        )
        K_n, I_n = 2 * K_n + I_n, K_n + I_n

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Bu özyineleme, modelin altın oranla kurduğu kuramsal bağlantıyı "
        "göstermek için kullanılır; tek başına görüntü ölçümünün doğruluğunu göstermez."
    )

    st.markdown("#### Geometrik anlatım çizimleri")
    g1, g2, g3 = st.columns(3)

    with g1:
        st.pyplot(fig_schematic_diffraction(), use_container_width=True)
        st.caption(
            "Kodla üretilmiş şematik beş katlı açısal örüntüdür; "
            "deneysel kırınım fotoğrafı değildir."
        )

    with g2:
        st.pyplot(fig_fivefold_rosette(), use_container_width=True)
        st.caption(
            "Genel beş katlı yıldız şemasıdır; gerçek Penrose döşemesinin kopyası değildir."
        )

    with g3:
        st.pyplot(fig_girih_schematic(), use_container_width=True)
        st.caption(
            "Girih esintili şematik on katlı motiftir; tarihî bir desenin "
            "birebir örneği değildir."
        )

    st.caption(
        "Kuasikristal keşfi Dan Shechtman ile ilişkilidir; yukarıdaki "
        "çizimlerin hiçbiri onun deneysel görüntüsü değildir."
    )


# ---------------------------------------------------------------------
# 2. KALİBRASYON VE SENTETİK ÖRNEK
# ---------------------------------------------------------------------

with tabs[1]:
    st.markdown("### Sentetik örnek ölçümü")

    st.info(
        "Bu bölüm sentetik örnekleri ve hesaplamanın teknik davranışını gösterir. "
        "Gerçek görüntülerde doğruluk veya klinik geçerlik testi değildir."
    )

    sample_name = st.selectbox(
        "İncelenecek sentetik örnek",
        options=list(SENTETIK_ORNEKLER.keys()),
        index=2,
    )
    sample_kind, recommended_k = SENTETIK_ORNEKLER[sample_name]

    selected_k = st.selectbox(
        "Bu örnekte kullanılacak dönel mertebe",
        options=list(K_DEGERLERI),
        index=list(K_DEGERLERI).index(recommended_k),
        format_func=lambda value: K_SECENEKLERI[value],
    )

    sample_image = sentetik_gorsel(sample_kind, seed=42)
    sample_result = gorsel_analiz(
        sample_image,
        k=selected_k,
        n_angles=n_angles,
        step_px=step_px,
        center_adjust=False,
        with_heat=True,
    )

    if "beş katlı" in sample_name.casefold():
        st.caption(
            "Bu örnek cos(5θ) ile üretilmiş sentetik beş katlı bir motiftir; "
            "gerçek Penrose döşemesi değildir. Varsayılan analiz mertebesi k=5'tir."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eşleşme oranı Sₖ", fmt_number(sample_result["S"]))
    m2.metric("Görüntüye ait B(p)", fmt_number(sample_result["B_chance"]))
    m3.metric("Normalize ölçüm Uₖ", fmt_number(sample_result["U"]))
    m4.metric("Geçerli çift kapsaması", f"{sample_result['coverage']:.1%}")

    img_col, binary_col, heat_col = st.columns(3)

    with img_col:
        st.image(
            sample_result["rgb"],
            caption="Sentetik örnek",
            use_container_width=True,
        )

    with binary_col:
        st.image(
            sample_result["binary"] * 255,
            caption="Ortalama parlaklık eşiğiyle ikileştirme",
            use_container_width=True,
        )

    with heat_col:
        fig = fig_heat_overlay(
            sample_result["rgb"],
            sample_result["heatmap"],
            "Sentetik örneğin yerel U haritası",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Sabit ayarlı teknik 180° örnek kontrolü")
    sym_image = exact_synthetic_test_image("symmetric")
    asym_image = exact_synthetic_test_image("asymmetric")

    sym_result = gorsel_analiz(
        sym_image,
        k=2,
        n_angles=180,
        step_px=4,
        center_adjust=False,
        with_heat=False,
    )
    asym_result = gorsel_analiz(
        asym_image,
        k=2,
        n_angles=180,
        step_px=4,
        center_adjust=False,
        with_heat=False,
    )

    test_col1, test_col2 = st.columns(2)

    with test_col1:
        st.image(
            resize_rgb(sym_image),
            caption="Gürültüsüz sentetik 180° dönel şekil",
            use_container_width=True,
        )
        st.metric("Sentetik simetrik şeklin U değeri", fmt_number(sym_result["U"]))

    with test_col2:
        st.image(
            resize_rgb(asym_image),
            caption="Gürültüsüz sentetik asimetrik şekil",
            use_container_width=True,
        )
        st.metric("Sentetik asimetrik şeklin U değeri", fmt_number(asym_result["U"]))

    st.caption(
        "Bu iki sentetik örnek yalnızca sabit hesaplamanın kontrollü şekillerdeki "
        "davranışını gösterir. HAM10000 sonuçlarını doğrulamaz."
    )


# ---------------------------------------------------------------------
# 3. GÖRSEL ANALİZ VE ISI HARİTASI
# ---------------------------------------------------------------------

with tabs[2]:
    st.markdown("### Bir veya iki görseli ölç")

    st.markdown(
        f"""
        <div class="section-card">
        Seçili ölçüm: <b>{K_SECENEKLERI[k]}</b> ·
        <b>{hassasiyet}</b> örnekleme ·
        Merkez ayarı: <b>{"açık" if center_adjust else "kapalı"}</b>.
        Her görsel için B(p) ayrı hesaplanır.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Sonuç, görüntüdeki dönel eşleşme ölçümünü açıklar; tek görsele "
        "iyi huylu/kötü huylu etiketi vermez."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        uploaded_a = st.file_uploader(
            "1. görsel (isteğe bağlı)",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_image_a",
        )

    with input_col2:
        uploaded_b = st.file_uploader(
            "2. görsel — ölçüm karşılaştırması için (isteğe bağlı)",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_image_b",
        )

    if uploaded_a is None:
        image_a = sentetik_gorsel("fivefold", seed=42)
        source_a = "Otomatik gösterilen sentetik beş katlı örnek"
    else:
        try:
            image_a = image_from_bytes(uploaded_a.getvalue())
            source_a = uploaded_a.name
        except Exception as exc:
            image_a = None
            source_a = ""
            st.error(f"1. görsel açılamadı: {exc}")

    if uploaded_b is not None:
        try:
            image_b = image_from_bytes(uploaded_b.getvalue())
            source_b = uploaded_b.name
        except Exception as exc:
            image_b = None
            source_b = ""
            st.error(f"2. görsel açılamadı: {exc}")
    else:
        image_b = None
        source_b = ""

    if image_a is not None:
        result_a = gorsel_analiz(
            image_a,
            k=k,
            n_angles=n_angles,
            step_px=step_px,
            center_adjust=center_adjust,
            with_heat=True,
        )

        if image_b is not None:
            result_b = gorsel_analiz(
                image_b,
                k=k,
                n_angles=n_angles,
                step_px=step_px,
                center_adjust=center_adjust,
                with_heat=True,
            )

            st.markdown("#### Yan yana ölçüm")
            col_a, col_b = st.columns(2)

            with col_a:
                panel_a = render_visual_panel(
                    f"Görsel 1: {source_a}",
                    result_a,
                )

            with col_b:
                panel_b = render_visual_panel(
                    f"Görsel 2: {source_b}",
                    result_b,
                )

            compare_single_results(
                result_a,
                panel_a,
                result_b,
                panel_b,
            )
        else:
            st.markdown(f"#### İncelenen görsel: {source_a}")
            render_visual_panel("Görsel 1", result_a)

    st.caption(
        "Isı haritasında kırmızı tonlar yerel U değerinin B(p) tabanının "
        "üzerinde; mavi tonlar altında olduğu bölgeleri gösterir. Boş alanlar "
        "geçerli yerel ölçüm bulunmayan yerlerdir."
    )


# ---------------------------------------------------------------------
# 4. ENDÜSTRİYEL VE GEOMETRİK İNCELEME
# ---------------------------------------------------------------------

with tabs[3]:
    st.markdown("### Prototiplerde olası simetri incelemesi")

    st.markdown(
        """
        <div class="section-card">
        Bu bölüm, simetrik tasarlanmış bir referans ile prototip görüntüsünün
        ölçüm farkını inceleme fikrini gösterir. Isı haritası yalnızca hesaplanan
        yerel U değerlerini gösterir; üretim onayı veya kusur tespiti değildir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    industry_k = st.selectbox(
        "Prototipte karşılaştırılacak dönel mertebe",
        options=list(K_DEGERLERI),
        index=list(K_DEGERLERI).index(5),
        format_func=lambda value: K_SECENEKLERI[value],
        key="industry_k",
    )

    disturbance = st.slider(
        "Sentetik örnek çiftindeki biçim değişikliği",
        min_value=0,
        max_value=40,
        value=12,
        step=1,
        help="Bu ayar yalnızca ekrandaki sentetik prototip örneğini değiştirir.",
    )

    ref_upload_col, prototype_upload_col = st.columns(2)

    with ref_upload_col:
        reference_upload = st.file_uploader(
            "Referans tasarım görseli (isteğe bağlı)",
            type=["jpg", "jpeg", "png", "webp"],
            key="industry_reference",
        )

    with prototype_upload_col:
        prototype_upload = st.file_uploader(
            "Prototip görseli (isteğe bağlı)",
            type=["jpg", "jpeg", "png", "webp"],
            key="industry_prototype",
        )

    use_uploaded_pair = (
        reference_upload is not None
        and prototype_upload is not None
    )

    if use_uploaded_pair:
        try:
            reference_image = image_from_bytes(reference_upload.getvalue())
            prototype_image = image_from_bytes(prototype_upload.getvalue())
            reference_name = reference_upload.name
            prototype_name = prototype_upload.name
        except Exception as exc:
            st.error(f"Yüklenen görseller açılamadı: {exc}")
            reference_image = sentetik_gorsel("fivefold", seed=42)
            prototype_image = sentetik_gorsel(
                "fivefold",
                disturbance=disturbance,
                seed=42,
            )
            reference_name = "Sentetik referans örneği"
            prototype_name = "Sentetik prototip örneği"
    else:
        if (reference_upload is None) != (prototype_upload is None):
            st.info(
                "Karşılaştırma için iki görsel de yüklenmelidir. "
                "Eksik çift yerine sentetik örnek gösteriliyor."
            )

        reference_image = sentetik_gorsel("fivefold", seed=42)
        prototype_image = sentetik_gorsel(
            "fivefold",
            disturbance=disturbance,
            seed=42,
        )
        reference_name = "Sentetik beş katlı referans"
        prototype_name = "Sentetik biçim değişikliği uygulanmış örnek"

    reference_result = gorsel_analiz(
        reference_image,
        k=industry_k,
        n_angles=n_angles,
        step_px=step_px,
        center_adjust=False,
        with_heat=False,
    )
    prototype_result = gorsel_analiz(
        prototype_image,
        k=industry_k,
        n_angles=n_angles,
        step_px=step_px,
        center_adjust=False,
        with_heat=True,
    )

    industry_col1, industry_col2 = st.columns(2)

    with industry_col1:
        st.write(f"**Referans: {reference_name}**")
        st.image(reference_result["rgb"], use_container_width=True)
        st.metric("Referans Uₖ", fmt_number(reference_result["U"]))
        st.caption(
            f"Sₖ={fmt_number(reference_result['S'])} · "
            f"B(p)={fmt_number(reference_result['B_chance'])}"
        )

    with industry_col2:
        st.write(f"**Prototip: {prototype_name}**")
        st.image(prototype_result["rgb"], use_container_width=True)
        st.metric("Prototip Uₖ", fmt_number(prototype_result["U"]))
        st.caption(
            f"Sₖ={fmt_number(prototype_result['S'])} · "
            f"B(p)={fmt_number(prototype_result['B_chance'])}"
        )

    delta_u = prototype_result["U"] - reference_result["U"]
    st.metric(
        "Ölçülen fark (prototip Uₖ − referans Uₖ)",
        fmt_number(delta_u),
    )

    if np.isfinite(delta_u):
        st.info(
            f"Bu ayarlarda prototip U değeri referans U değerinden "
            f"{fmt_number(delta_u)} farklıdır. Bu, yalnızca iki görüntünün "
            "ölçüm farkıdır; üretim kusuru veya uygunluk kararı değildir."
        )

    heat_col, note_col = st.columns([1.1, 1])

    with heat_col:
        fig = fig_industry_overlay(
            prototype_result["rgb"],
            prototype_result["heatmap"],
            "Prototip örneğinin yerel U haritası",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with note_col:
        st.markdown(
            """
            #### Bu prototip neyi gösteriyor?

            - Referans ve prototip aynı dönel mertebede ölçülür.
            - İki görüntünün U ölçümü ve aralarındaki fark raporlanır.
            - Sıcak/soğuk bölgeler hesaplanan yerel U değerlerini gösterir.
            - Gerçek üretim toleransı, standart görüntü alma koşulları ve
              kabul/red sınırları bu prototipte belirlenmiş değildir.
            """
        )
        st.caption(
            "Varsayılan sentetik örnek beş katlıdır; başlangıç mertebesi k=5'tir."
        )


# ---------------------------------------------------------------------
# 5. HAM10000 GRUP KARŞILAŞTIRMASI
# ---------------------------------------------------------------------

with tabs[4]:
    st.markdown("### İyi huylu nevüs ve kötü huylu melanom etiketli görseller")

    st.info(
        f"**{GROUP_NV}** ile **{GROUP_MEL}** ayrı gruplar olarak karşılaştırılır. "
        "Bunlar veri setindeki etiketlerdir; tek görsel sınıflandırması değildir."
    )

    st.markdown(
        f"""
        <div class="section-card">
        <b>Önceden belirlenen ana protokol:</b> 100 nv + 100 mel görsel,
        k=2 (180° dönel karşılaştırma), Hassas örnekleme (180 açı, 4 px adım),
        merkez ayarı kapalı ve her görsel için ayrı B(p).
        Üç ana ölçüt için Bonferroni eşiği α={ALFA_DUZELTILMIS:.4f}'tür.
        <br><br>
        100+100 dışındaki yüklemeler analiz edilebilir; ancak ana 100+100
        protokol sonucu yerine geçmez ve ayrı/ikincil inceleme olarak
        yorumlanmalıdır.
        </div>
        """,
        unsafe_allow_html=True,
    )

    nv_upload_col, mel_upload_col = st.columns(2)

    with nv_upload_col:
        nv_files = st.file_uploader(
            f"{GROUP_NV} — JPG / PNG",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_nv",
        )

    with mel_upload_col:
        mel_files = st.file_uploader(
            f"{GROUP_MEL} — JPG / PNG",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_mel",
        )

    metadata_upload = st.file_uploader(
        "İsteğe bağlı: HAM10000 metadata CSV "
        "(image_id, lesion_id, dx sütunları)",
        type=["csv"],
        key="batch_metadata",
    )

    reference_mask_files = st.file_uploader(
        "İsteğe bağlı kalite kontrolü: elle çizilmiş siyah-beyaz referans maskeler "
        "(dosya adı görselin image_id adıyla aynı olmalı)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="batch_reference_masks",
    )

    nv_records = uploaded_records(nv_files)
    mel_records = uploaded_records(mel_files)
    reference_records = uploaded_records(reference_mask_files)

    metadata_bytes = (
        metadata_upload.getvalue()
        if metadata_upload is not None
        else None
    )

    duplicate_issues = duplicate_report(
        {
            GROUP_NV: nv_records,
            GROUP_MEL: mel_records,
        }
    )
    duplicate_mask_ids = duplicate_reference_names(reference_records)
    mask_lookup = reference_mask_map(reference_records)

    meta_check = check_metadata(
        metadata_bytes,
        nv_records,
        mel_records,
    )

    count_col1, count_col2, count_col3 = st.columns(3)
    count_col1.metric("İyi huylu nevüs (nv) dosyası", len(nv_records))
    count_col2.metric("Kötü huylu melanom (mel) dosyası", len(mel_records))
    count_col3.metric(
        "100 / 100 yükleme sayısı",
        (
            "Sağlandı"
            if len(nv_records) == 100 and len(mel_records) == 100
            else "Sağlanmadı"
        ),
    )

    if duplicate_issues:
        st.error(
            "Görseller arasında tekrarlanan içerik veya dosya adı bulundu. "
            "Tekrarları temizleyip yeniden yükleyin; analiz başlatılmaz."
        )
        with st.expander("Görsel tekrar denetimi"):
            for issue in duplicate_issues:
                st.write("• " + issue)

    if duplicate_mask_ids:
        st.warning(
            "Aynı görsele ait birden fazla referans maske adı bulundu. "
            "Bu adlar belirsiz olduğu için ilgili maskeler kalite "
            "karşılaştırmasında kullanılmayacak."
        )

    if metadata_bytes is None:
        st.warning(meta_check["messages"][0])
    else:
        if meta_check["valid"]:
            st.success(
                "Metadata ile image_id/dx eşleşmesi ve lesion_id tekilliği kontrol edildi."
            )
        else:
            st.warning(
                "Metadata denetimi tüm dosyalar için geçerli eşleşme sağlayamadı."
            )

        with st.expander("Metadata denetimi ayrıntıları"):
            for message in meta_check["messages"]:
                st.write("• " + message)

    st.caption(
        "Metadata denetimi, tohum 42 ile örnek seçimini tek başına doğrulamaz. "
        "Elle çizilmiş referans maskeler siyah arka plan ve beyaz lezyon alanı "
        "şeklinde hazırlanmalı; dosya adları görsellerin image_id adıyla eşleşmelidir."
    )

    if reference_records:
        st.caption(
            f"Referans maskesi dosyası: {len(reference_records)}. "
            "Yalnızca dosya adı görselin image_id adıyla eşleşen maskeler karşılaştırılır."
        )

    can_run = (
        len(nv_records) >= 2
        and len(mel_records) >= 2
        and not duplicate_issues
    )

    signature = batch_signature(
        nv_records,
        mel_records,
        metadata_bytes,
        reference_records,
    )

    run_button = st.button(
        "HAM10000 grup analizini çalıştır",
        disabled=not can_run,
        key="run_batch",
    )

    if not can_run:
        st.caption(
            "Analiz için her grupta en az iki görsel yükleyin ve görsel tekrarlarını giderin."
        )

    if run_button:
        with st.spinner(
            "Görseller işleniyor; ana A/B/C ölçümleri için sabit ayarlar uygulanıyor..."
        ):
            batch_result = run_batch(
                nv_records,
                mel_records,
                image_info=meta_check.get("image_info", {}),
                mask_lookup=mask_lookup,
            )
            batch_result["signature"] = signature
            batch_result["metadata_valid"] = meta_check["valid"]
            batch_result["unique_nv_lesions"] = meta_check["unique_nv_lesions"]
            batch_result["unique_mel_lesions"] = meta_check["unique_mel_lesions"]
            batch_result["uploaded_nv"] = len(nv_records)
            batch_result["uploaded_mel"] = len(mel_records)
            batch_result["reference_masks_uploaded"] = len(reference_records)
            st.session_state["ysum_batch_result"] = batch_result

    saved_batch = st.session_state.get("ysum_batch_result")

    if saved_batch is not None and saved_batch.get("signature") != signature:
        st.info(
            "Yüklenen dosyalar değişti. Güncel sonuç için analiz düğmesine yeniden basın."
        )

    if saved_batch is not None and saved_batch.get("signature") == signature:
        batch_data = saved_batch["data"]
        errors = saved_batch["errors"]

        if batch_data.empty:
            st.error("İşlenebilir görsel bulunamadı.")
        else:
            required_columns = [
                "lesion_id",
                "A_mask_U",
                "mirror_asym_lr",
                "mirror_asym_tb",
                "segmentasyon_gecerli",
                "lezyon_alan_orani",
                "p_acik_piksel",
                "referans_maskesi",
                "referans_Dice",
                "referans_IoU",
            ]
            for column in required_columns:
                if column not in batch_data.columns:
                    batch_data[column] = np.nan

            processed_nv = int(saved_batch["nv_processed"])
            processed_mel = int(saved_batch["mel_processed"])
            no_errors = len(errors) == 0
            metadata_ok = bool(saved_batch.get("metadata_valid", False))
            is_primary_count = processed_nv == 100 and processed_mel == 100

            st.markdown("#### Yükleme, örneklem ve protokol durumu")

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("İşlenen nv görseli", processed_nv)
            pcol2.metric("İşlenen mel görseli", processed_mel)
            pcol3.metric(
                "Metadata ile benzersiz nv lezyonu",
                (
                    saved_batch.get("unique_nv_lesions", 0)
                    if metadata_ok
                    else "Kontrol edilmedi"
                ),
            )
            pcol4.metric(
                "Metadata ile benzersiz mel lezyonu",
                (
                    saved_batch.get("unique_mel_lesions", 0)
                    if metadata_ok
                    else "Kontrol edilmedi"
                ),
            )

            if is_primary_count and no_errors and metadata_ok:
                st.success(
                    "100 iyi huylu nevüs (nv) ve 100 kötü huylu melanom (mel) "
                    "görseli işlendi; metadata etiketleri ve lesion_id tekilliği kontrol edildi."
                )
                st.caption(
                    "Tohum 42 ile örnek seçimi ayrıca saklanmalıdır; uygulama bunu "
                    "metadata üzerinden bağımsız olarak doğrulayamaz."
                )
            elif processed_nv > 100 or processed_mel > 100:
                st.warning(
                    "Bu yükleme 100+100 ana protokolden daha büyüktür. "
                    "Sonuçlar genişletilmiş/ikincil analiz olarak raporlanmalı; "
                    "önceden belirlenen 100+100 ana sonuçla karıştırılmamalıdır."
                )
                if not metadata_ok:
                    st.warning(
                        "Metadata ve benzersiz lesion_id kontrolü geçerli olmadığı "
                        "için bağımsız örneklem varsayımı ayrıca doğrulanamadı."
                    )
            else:
                st.warning(
                    "Bu yükleme 100/100 ana protokol koşulunu karşılamıyor veya "
                    "bazı dosyalar işlenemedi. Sonuçları ön inceleme olarak değerlendirin."
                )

            if not metadata_ok:
                st.caption(
                    "Metadata denetimi geçerli değilse aynı lezyona ait birden fazla "
                    "görsel bulunabilir. Bu durumda bağımsızlık varsayımına dayanan "
                    "p değerleri ihtiyatla yorumlanmalıdır."
                )

            if errors:
                with st.expander(f"Okunamayan dosyalar ({len(errors)})"):
                    for error in errors:
                        st.write("• " + error)

            st.markdown("#### Ana A/B/C ölçütlerinin grup dağılımı")
            fig = fig_batch_boxplots(batch_data)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown(
                """
                **A, B ve C ayrı ana ölçütlerdir; birleştirilmiş puan oluşturulmaz.**
                `B(p)` şans düzeyi, `B` ise kenar düzensizliği ölçütüdür.
                """
            )

            batch_stats = build_batch_statistics(batch_data)
            stats_df = results_table(batch_stats)

            st.markdown("#### Ana A/B/C istatistiksel karşılaştırması")
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
            )

            for item in batch_stats:
                st.markdown(
                    f"""
                    <div class="result-card">
                    <b>{html.escape(item['metric'])}</b><br>
                    {html.escape(item['comment'])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Ana A/B/C testlerinde önceden belirlenen yönlü Welch testleri "
                f"ve Bonferroni eşiği α={ALFA_DUZELTILMIS:.4f} kullanılır. "
                "Gösterilen olağan %95 güven aralıkları çoklu test düzeltmesi içermez. "
                "AUC grup etiketlerini tanı doğruluğunda sınıflandıran bir sonuç değildir."
            )

            st.markdown("#### Segmentasyon ve ölçüm kalite özeti")
            quality_df = batch_quality_summary(batch_data)
            st.dataframe(
                quality_df,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "`p_acik_piksel`, A ölçümünde ortalama parlaklık eşiğiyle "
                "ikileştirilmiş tüm görüntünün açık piksel oranıdır; arka plan "
                "oranı olarak adlandırılmamalıdır."
            )

            if reference_records:
                matched_reference_count = int(
                    batch_data["referans_Dice"].notna().sum()
                )
                if matched_reference_count:
                    st.success(
                        f"{matched_reference_count} görsel için yüklenen elle "
                        "çizilmiş referans maskesiyle Dice/IoU karşılaştırması yapıldı."
                    )
                else:
                    st.warning(
                        "Yüklenen referans maskelerinden hiçbiri görsel dosya adıyla "
                        "eşleşmedi; Dice/IoU hesaplanamadı."
                    )
            else:
                st.info(
                    "Otsu maskesinin doğruluğunu sayısal olarak değerlendirmek için "
                    "elle çizilmiş referans maskeler yüklenmedi. Otomatik "
                    "segmentasyonun geçerli görünmesi, maskenin doğru olduğunu kanıtlamaz."
                )

            review_sample = make_mask_review_sample(
                batch_data,
                per_group=25,
                seed=42,
            )
            if not review_sample.empty:
                st.download_button(
                    "Elle maske kalite kontrolü için örnek listesini indir",
                    data=review_sample.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ysum_maske_kalite_kontrol_ornek_listesi.csv",
                    mime="text/csv",
                    help=(
                        "Bu liste grup başına en fazla 25 görsel içerir. "
                        "Ana analiz örneklemini değiştirmez."
                    ),
                )
                st.caption(
                    "İsterseniz bu listeden her grupta en fazla 25 görsel için "
                    "elle maske hazırlayıp adları aynı image_id ile yeniden "
                    "yükleyebilirsiniz. Bu kontrol ana analiz örneklemini değiştirmez."
                )

            st.markdown("#### Keşifsel lezyon-maskesi dönel ölçümü")
            st.info(
                "Bu ölçüm Otsu maskesinin şekli üzerinde hesaplanır. Ana A ölçümü "
                "tüm görüntüden hesaplanır; maske dönel U onun yerine geçmez ve "
                "ana A/B/C p değerlerine eklenmez."
            )

            mask_rotation_df = build_mask_rotation_summary(batch_data)
            st.dataframe(
                mask_rotation_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Keşifsel lezyon-maskesi ayna-asimetrisi")
            st.info(
                "Sol-sağ ve üst-alt yansımaları, Otsu maskesinin kendi merkezinden "
                "karşılaştırılır. Bunlar ana A/B/C sonuçlarından ayrı keşifsel "
                "ölçümlerdir; ABCD kuralının tam uygulaması değildir."
            )

            fig = fig_mirror_boxplots(batch_data)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            mirror_stats = build_mirror_statistics(batch_data)
            st.dataframe(
                mirror_results_table(mirror_stats),
                use_container_width=True,
                hide_index=True,
            )

            for item in mirror_stats:
                st.markdown(
                    f"""
                    <div class="result-card">
                    <b>{html.escape(item['axis_label'])}</b><br>
                    {html.escape(item['comment'])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Ayna ölçümleri için iki yönlü Mann–Whitney U ve iki eksen arasında "
                "Holm düzeltmesi gösterilir. Test bağımsız gözlemler gerektirir; "
                "dağılım şekilleri farklıysa yalnızca medyan farkı gibi "
                "yorumlanmamalıdır. Keşifsel sonuçlar ana A/B/C sonuçlarını değiştirmez."
            )

            display_columns = [
                "grup",
                "dosya",
                "lesion_id",
                "A_U",
                "A_S",
                "B_p",
                "B_edge",
                "C_colors",
                "renk_siniflari",
                "A_mask_U",
                "mirror_asym_lr",
                "mirror_asym_tb",
                "kapsama",
                "segmentasyon_gecerli",
                "lezyon_alan_orani",
                "p_acik_piksel",
                "referans_maskesi",
                "referans_Dice",
                "referans_IoU",
            ]

            rename_columns = {
                "grup": "Veri setindeki grup etiketi",
                "dosya": "Dosya",
                "lesion_id": "Metadata lesion_id",
                "A_U": "A — Tüm görüntüde dönel U",
                "A_S": "A — Eşleşme oranı S",
                "B_p": "A — Görüntüye ait B(p)",
                "B_edge": "B — Kenar düzensizliği",
                "C_colors": "C — Renk sınıfı sayısı",
                "renk_siniflari": "Bulunan renk sınıfları",
                "A_mask_U": "Keşifsel — Maske dönel U",
                "mirror_asym_lr": "Keşifsel — Sol-sağ maske asimetrisi",
                "mirror_asym_tb": "Keşifsel — Üst-alt maske asimetrisi",
                "kapsama": "Geçerli çift kapsaması",
                "segmentasyon_gecerli": "Otsu segmentasyonu geçerli",
                "lezyon_alan_orani": "Otsu lezyon alan oranı",
                "p_acik_piksel": "A için açık piksel oranı p",
                "referans_maskesi": "Eşleşen elle çizilmiş referans maskesi",
                "referans_Dice": "Referans maskeye Dice",
                "referans_IoU": "Referans maskeye IoU",
            }

            st.markdown("#### Görsel başına ölçüm tablosu")
            st.dataframe(
                batch_data[display_columns].rename(columns=rename_columns),
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = batch_data.to_csv(index=False).encode("utf-8-sig")
            txt_lines = [
                "YSÜM — HAM10000 grup karşılaştırması",
                "",
                f"İyi huylu nevüs (nv) işlenen görsel: {processed_nv}",
                f"Kötü huylu melanom (mel) işlenen görsel: {processed_mel}",
                f"100/100 işlenen görsel koşulu: {'Evet' if is_primary_count else 'Hayır'}",
                f"Metadata lesion_id denetimi: {'Geçerli' if metadata_ok else 'Geçerli değil / yapılmadı'}",
                "Ana ölçüm ayarları: k=2, 180 açı, 4 px adım, merkez ayarı kapalı.",
                "B(p) her görüntü için ayrı hesaplanmıştır.",
                f"A/B/C için Bonferroni eşiği: {ALFA_DUZELTILMIS:.4f}",
                "AUC sıralama ölçüsüdür; p değerinin yerine geçmez ve tanı doğruluğu değildir.",
                "",
                "Ana A/B/C yorumları:",
            ]

            for item in batch_stats:
                txt_lines.append(f"- {item['metric']}: {item['comment']}")

            txt_lines.extend(["", "Keşifsel ayna-asimetrisi yorumları:"])
            for item in mirror_stats:
                txt_lines.append(
                    f"- {item['axis_label']}: {item['comment']}"
                )

            txt_lines.extend(
                [
                    "",
                    "Segmentasyon ve ölçüm kalite özeti:",
                    quality_df.to_string(index=False),
                    "",
                    "Not: Anlamlı fark bulunmaması gruplar arasında hiç fark olmadığı anlamına gelmez.",
                    "Metadata tohum 42 ile örnek seçimini bağımsız doğrulamaz.",
                    "Maske dönel U ve ayna-asimetrisi ölçümleri keşifseldir.",
                    f"Uygulama sürümü: {SURUM}",
                ]
            )

            txt_bytes = "\n".join(txt_lines).encode("utf-8")

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                st.download_button(
                    "Ölçüm tablosunu CSV indir",
                    data=csv_bytes,
                    file_name="ysum_ham10000_olcumleri.csv",
                    mime="text/csv",
                )

            with download_col2:
                st.download_button(
                    "Sonuç özetini TXT indir",
                    data=txt_bytes,
                    file_name="ysum_ham10000_sonuc_ozeti.txt",
                    mime="text/plain",
                )


# ---------------------------------------------------------------------
# 6. MODEL MİMARİSİ VE METODOLOJİ
# ---------------------------------------------------------------------

with tabs[5]:
    st.markdown("### Model mimarisi ve ölçüm adımları")
    st.warning(
        "Bu uygulama araştırma/prototip amaçlıdır. Grup farkları tek başına "
        "model doğrulaması, klinik performans veya tanı başarısı göstermez."
    )

    st.markdown(
        """
        <div class="section-card">
        YSÜM, görüntülerden dönel benzerlik ölçümü üretmek ve sonuçları ayrı
        görsel ölçütlerle incelemek için hazırlanmış bir proje prototipidir.
        Penrose/altın oran bağlantısı kuramsal bir referanstır; A, B ve C
        ölçümlerinin tümü bu matematikten türetilmiş değildir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### A — Ana dönel simetri ölçümü")
    st.markdown(
        """
        1. Görsel RGB biçimine çevrilip **200×200** boyutuna getirilir.
        2. Gri tonlama yapılır ve tüm görüntünün ortalama parlaklık eşiğiyle
           ikili görüntü oluşturulur.
        3. Merkez çevresinde seçilen açılarda ve yarıçaplarda,
           aralarında `360°/k` bulunan piksel çiftleri karşılaştırılır.
        4. Her açısal örnekte geçerli çift kapsaması en az **0,50** olmalıdır.
           Tamamen tek tip örnek çiftleri ölçüme katılmaz.
        5. Eşleşme oranı `Sₖ`, görüntünün kendi açık piksel oranından
           hesaplanan `B(p)=p²+(1−p)²` ve normalize `Uₖ` raporlanır.
        6. Ana A ölçümü tüm görüntü üzerinden hesaplanır; lezyon maskesi,
           arka plan, kadraj ve aydınlatmanın etkisi olabilir.
        """
    )

    st.latex(r"U_k=\frac{S_k-B(p)}{1-B(p)}")

    st.markdown("#### Dönel mertebe neyi ifade eder?")
    st.markdown(
        """
        - **k=2:** 180° dönel karşılaştırma; ayna yansıması/eksen simetrisi değildir.
        - **k=5:** 72° aralıklı dönel karşılaştırma; Penrose döşemesi veya
          kuasikristal olduğu sonucunu tek başına vermez.
        - **k=6 ve k=8:** sırasıyla 60° ve 45° dönel karşılaştırmadır.
        """
    )

    st.markdown("#### B ve C — Ana ek görüntü ölçütleri")
    st.markdown(
        """
        - **B — Kenar düzensizliği:** Otsu ile seçilen maskenin çevre²/(4π×alan)
          indeksidir; maske hatası bu ölçümü etkileyebilir.
        - **C — Renk sınıfı sayısı:** sabit altı renk paletinde, lezyon maskesinin
          en az %5'ini oluşturan renk sınıflarının sayısıdır.
        - `B(p)` A ölçümündeki şans eşleşme düzeyidir; `B` kenar düzensizliği
          ölçütüyle aynı şey değildir.
        """
    )

    st.markdown("#### Otsu segmentasyonu ve kalite kontrolü")
    st.markdown(
        """
        Otsu maskesi otomatik bir yaklaşık bölütlemedir. Uygulamadaki
        “segmentasyon geçerli” ifadesi yalnızca alan oranı koşulunun sağlandığını
        söyler; sınırın lezyonu doğru çizdiğini kanıtlamaz. Kaliteyi ölçmek için
        bazı görsellerde elle çizilmiş referans maskelerle Dice ve IoU
        karşılaştırması yapılabilir. Referans maskede lezyon beyaz,
        arka plan siyah olmalıdır.
        """
    )

    st.markdown("#### Keşifsel maske ve ayna ölçümleri")
    st.markdown(
        """
        - **Maske dönel U:** Otsu maskesi üzerinde ayrı hesaplanır; tüm görüntüden
          hesaplanan ana A ölçümünün yerine geçmez. Toplu bölümde betimsel olarak gösterilir.
        - **Sol-sağ ve üst-alt ayna asimetrisi:** Otsu maskesinin kendi merkezinden
          yansıtılmasıyla hesaplanır. Renk asimetrisi değildir ve ABCD kuralının
          tam uygulaması değildir.
        - Bu ölçümler keşifseldir; ana A/B/C hipotezlerine eklenmez.
        """
    )

    st.markdown("#### İstatistiksel karşılaştırma ve AUC")
    st.markdown(
        f"""
        Ana grup karşılaştırmalarında beklenen yönlü Welch testleri kullanılır:
        A için iyi huylu nevüs (nv) > kötü huylu melanom (mel);
        B ve C için kötü huylu melanom (mel) > iyi huylu nevüs (nv).
        Üç ana ölçüt için Bonferroni eşiği α={ALFA_DUZELTILMIS:.4f}'tür.
        AUC, beklenen grup yönündeki sıralama ölçüsüdür; p değerinin yerine
        geçmez ve klinik tanı doğruluğu olarak yorumlanmaz.
        """
    )

    st.markdown("#### Örneklem ve bağımsızlık")
    st.markdown(
        """
        HAM10000 metadata dosyası `image_id`, `lesion_id` ve `dx` sütunlarıyla
        dosya eşleşmesini ve lezyon kimliklerini kontrol etmekte kullanılır.
        Aynı lesion_id'ye ait tekrarlı görseller bağımsız lezyon örnekleri
        olarak sayılmamalıdır. Metadata, örneklerin tohum 42 ile seçildiğini
        tek başına doğrulamaz. 100+100 dışındaki analizler ana protokolden
        ayrı raporlanmalıdır.
        """
    )

    st.markdown("#### Kısa kaynak notu")
    st.markdown(
        """
        - Penrose, R. (1974). *The role of aesthetics in pure and applied mathematical research.*
        - Shechtman, D. ve ark. (1984). *Metallic phase with long-range orientational order and no translational symmetry.*
        - Lu, P. J. & Steinhardt, P. J. (2007). *Decagonal and quasi-crystalline tilings in medieval Islamic architecture.*
        - Tschandl, P., Rosendahl, C. & Kittler, H. (2018). *The HAM10000 dataset.* Scientific Data, 5, 180161.
        """
    )

    st.caption(
        "HAM10000 için veri kümesi sürümünü ve indirme tarihini raporda "
        "belirtin; veri kümesinin lisans koşullarına uyun."
    )
```
