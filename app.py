

# YSÜM — Yerel Simetri Üstünlük Modeli
# Düzenlenmiş sürüm: giriş doğrulama, hata dayanıklılığı ve raporlama iyileştirmeleri.
# Ölçümler ayrı ayrı raporlanır; klinik tanı veya otomatik kabul/red kararı üretmez.

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


# ---------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="YSÜM | Görüntü Simetri İncelemesi",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SURUM = "YSÜM-v3.2"
IMG_N = 200

ALFA = 0.05
KRITER_SAYISI = 3
ALFA_DUZELTILMIS = ALFA / KRITER_SAYISI

MIN_KAPSAMA = 0.50
DUZ_ALANLARI_DISLA = True

YORUM_ESIGI = 0.15
SKOR_YARICAPLARI = (20, 40, 60, 80)
ISI_YARICAPLARI = (10, 20, 30, 40)
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
# YARDIMCI FONKSİYONLAR
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
        raise ValueError(f"Dosya geçerli veya desteklenen bir görsel değil: {exc}") from exc


def resize_rgb(image):
    """Görseli RGB biçiminde 200×200 boyutuna getirir."""
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            pil_image = Image.fromarray(np.uint8(np.clip(arr, 0, 255))).convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            mode = "RGBA" if arr.shape[2] == 4 else "RGB"
            pil_image = Image.fromarray(np.uint8(np.clip(arr, 0, 255)), mode=mode).convert("RGB")
        else:
            raise ValueError("Görsel dizisi H×W, H×W×3 veya H×W×4 biçiminde olmalıdır.")
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
    labels, count = ndimage.label(dark, structure=np.ones((3, 3), dtype=np.uint8))
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
    """Varsayılan merkez geometriktir; istenirse koyu merkez bileşenine göre ayarlanır."""
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
        return np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.uint8), valid

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
        valid.astype(float), size=window, mode="constant", cval=0.0
    )
    matches = ((binary == rotated) & valid).astype(float)
    match_fraction = ndimage.uniform_filter(
        matches, size=window, mode="constant", cval=0.0
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

    if step_px <= 0:
        step_px = 4

    radius_values = set(range(20, 81, int(step_px)))
    radius_values.update(SKOR_YARICAPLARI)
    radius_values = sorted(r for r in radius_values if 0 < r <= 80)

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

    if compared_count > 0:
        score_s = float(equal_count / compared_count)
    else:
        score_s = float("nan")

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
        heat, heat_coverage = local_u_map(
            binary,
            center,
            k,
            baseline_b,
        )
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


def lesion_features(rgb):
    """B kenar düzensizliği ve C renk sınıfı sayısı ek ölçümlerini hesaplar."""
    seg = segment_lesion(rgb)
    if not seg["valid"] or not np.any(seg["mask"]):
        return {
            "segmentation_valid": False,
            "area_fraction": seg["area_fraction"],
            "B_edge": float("nan"),
            "C_colors": float("nan"),
            "color_names": "",
        }

    mask = seg["mask"]
    area = float(np.count_nonzero(mask))

    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = mask & (~eroded)
    perimeter = float(np.count_nonzero(boundary))

    # Basit çevre²/(4π×alan) kenar düzensizliği indeksi.
    # Büyük değer daha uzun/düzensiz bir sınırla uyumludur.
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
        return {
            "segmentation_valid": False,
            "area_fraction": seg["area_fraction"],
            "B_edge": float("nan"),
            "C_colors": float("nan"),
            "color_names": "",
        }

    distances = np.sum(
        (pixels[:, None, :] - palette[None, :, :]) ** 2,
        axis=2,
    )
    class_ids = np.argmin(distances, axis=1)
    counts = np.bincount(class_ids, minlength=len(palette_names))
    fractions = counts / float(max(counts.sum(), 1))

    present = [
        palette_names[i]
        for i, fraction in enumerate(fractions)
        if fraction >= RENK_MIN_ORAN
    ]

    return {
        "segmentation_valid": True,
        "area_fraction": float(seg["area_fraction"]),
        "B_edge": float(edge_irregularity),
        "C_colors": float(len(present)),
        "color_names": ", ".join(present),
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
    """Kontrollü, matematiksel sentetik motif üretir; Penrose döşemesi değildir."""
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

    return Image.fromarray(background, mode="RGB")


def fig_penrose_baseline():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    p_ref = 1.0 / phi
    b_ref = p_ref**2 + (1.0 - p_ref) ** 2

    p = np.linspace(0.0, 1.0, 401)
    b = p**2 + (1.0 - p) ** 2

    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.plot(p, b, color="#365fa7", linewidth=3, label="B(p) = p² + (1−p)²")
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

    # Görsel anlatım için üretilmiş şematik örüntü; fiziksel simülasyon değildir.
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
    ax.fill(x, y, color="#68b7aa", alpha=0.82, edgecolor="#205e70", linewidth=2)
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
    for step, color, width in [(1, "#8d67a8", 1.2), (3, "#347f91", 1.8), (4, "#dc8d4c", 1.2)]:
        for i in range(10):
            j = (i + step) % 10
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
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


def fig_batch_boxplots(data):
    specs = [
        ("A — Dönel simetri U", "A_U"),
        ("B — Kenar düzensizliği", "B_edge"),
        ("C — Renk sınıfı sayısı", "C_colors"),
    ]

    groups = [
        (GROUP_NV, GROUP_NV_CHART, "#55a9a2"),
        (GROUP_MEL, GROUP_MEL_CHART, "#8b72bd"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (title, column) in zip(axes, specs):
        plot_data = []

        for group_name, label, color in groups:
            values = pd.to_numeric(
                data.loc[data["grup"] == group_name, column],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)

            values = values[np.isfinite(values)]
            plot_data.append((label, color, values))

        if any(len(values) == 0 for _, _, values in plot_data):
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

        for position, (label, color, values) in enumerate(plot_data, start=1):
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            iqr = q3 - q1

            lower_values = values[values >= q1 - 1.5 * iqr]
            upper_values = values[values <= q3 + 1.5 * iqr]

            whisker_low = (
                float(np.min(lower_values))
                if len(lower_values) > 0
                else float(np.min(values))
            )
            whisker_high = (
                float(np.max(upper_values))
                if len(upper_values) > 0
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

        ax.set_title(title)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [GROUP_NV_CHART, GROUP_MEL_CHART],
            rotation=12,
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.22)
        ax.margins(y=0.12)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# İSTATİSTİKSEL YARDIMCILAR
# ---------------------------------------------------------------------

def welch_directional(expected_high, expected_low):
    """
    Önceden belirlenmiş yön için Welch t testi.
    expected_high - expected_low farkı beklenen yönde pozitif olmalıdır.
    """
    x = pd.to_numeric(pd.Series(expected_high), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(expected_low), errors="coerce").dropna().to_numpy(dtype=float)

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

    if se_squared <= 0:
        if abs(difference) < 1e-15:
            t_value = 0.0
            p_one = 0.5
            p_two = 1.0
            df = float("inf")
            ci_low = 0.0
            ci_high = 0.0
        else:
            t_value = math.copysign(float("inf"), difference)
            p_one = 0.0 if difference > 0 else 1.0
            p_two = 0.0
            df = float("inf")
            ci_low = difference
            ci_high = difference
    else:
        standard_error = math.sqrt(se_squared)
        t_value = difference / standard_error

        denominator = 0.0
        if n1 > 1:
            denominator += (term_x * term_x) / (n1 - 1)
        if n2 > 1:
            denominator += (term_y * term_y) / (n2 - 1)

        df = (se_squared * se_squared / denominator) if denominator > 0 else float("inf")

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
            "difference": difference,
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


def fmt_number(value, digits=4):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def fmt_p(value):
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def make_comment(metric_name, high_group, low_group, result):
    if result["n_high"] < 2 or result["n_low"] < 2:
        return "Bu ölçüt için iki grupta da en az ikişer geçerli değer bulunmalıdır."

    difference = result["difference"]
    p_one = result["p_one"]

    if difference > 0 and p_one < ALFA_DUZELTILMIS:
        return (
            f"Bu örneklemde {metric_name}, {high_group} grubunda "
            f"{low_group} grubuna göre daha yüksek bulundu. "
            f"Beklenen yöndeki fark, üç ölçüt için Bonferroni düzeltmeli "
            f"eşikten (α={ALFA_DUZELTILMIS:.4f}) küçüktür."
        )

    if difference > 0:
        return (
            f"Bu örneklemde {metric_name} beklenen yönde daha yüksek olsa da "
            f"Bonferroni düzeltmeli eşikte istatistiksel olarak anlamlı değildir."
        )

    return (
        f"Bu örneklemde {metric_name} için beklenen yön desteklenmedi. "
        f"Bu sonuç, gruplar arasında hiçbir fark olmadığı anlamına gelmez."
    )


# ---------------------------------------------------------------------
# HAM10000 DOSYA VE METADATA DENETİMİ
# ---------------------------------------------------------------------

def uploaded_records(files):
    records = []
    for uploaded in files or []:
        try:
            data = uploaded.getvalue()
            if not data:
                continue
            records.append(
                {
                    "name": str(uploaded.name),
                    "bytes": data,
                    "md5": hashlib.md5(data).hexdigest(),
                    "normalized_name": Path(str(uploaded.name)).name.casefold(),
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
            current = (group_name, record["name"])

            if md5 in seen_md5:
                prior = seen_md5[md5]
                issues.append(
                    f"Aynı içerik MD5 ile tekrarlandı: {prior[1]} ve {record['name']}."
                )
            else:
                seen_md5[md5] = current

            if name in seen_names:
                prior = seen_names[name]
                issues.append(
                    f"Büyük/küçük harf duyarsız dosya adı tekrarı: "
                    f"{prior[1]} ve {record['name']}."
                )
            else:
                seen_names[name] = current

    return sorted(set(issues))


def normalize_image_id(value):
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).stem.casefold()


def check_metadata(metadata_bytes, nv_records, mel_records):
    result = {
        "provided": metadata_bytes is not None,
        "valid": False,
        "messages": [],
        "matched": 0,
    }

    if metadata_bytes is None:
        result["messages"].append(
            "Metadata CSV yüklenmedi; image_id, dx ve lesion_id eşleşmeleri "
            "uygulama içinde doğrulanamadı."
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

    column_lookup = {str(column).strip().casefold(): column for column in metadata.columns}
    required = ["image_id", "lesion_id", "dx"]
    missing = [column for column in required if column not in column_lookup]

    if missing:
        result["messages"].append(
            "Metadata CSV içinde gereken sütunlar bulunamadı: "
            + ", ".join(missing)
            + ". Gerekli sütunlar: image_id, lesion_id, dx."
        )
        return result

    image_col = column_lookup["image_id"]
    lesion_col = column_lookup["lesion_id"]
    dx_col = column_lookup["dx"]

    metadata = metadata.copy()
    metadata["_norm_image_id"] = metadata[image_col].map(normalize_image_id)
    metadata["_norm_lesion_id"] = metadata[lesion_col].astype(str).str.strip().str.casefold()
    metadata["_norm_dx"] = metadata[dx_col].astype(str).str.strip().str.casefold()

    duplicate_ids = metadata["_norm_image_id"].duplicated(keep=False)
    if duplicate_ids.any():
        result["messages"].append(
            "Metadata içinde tekrarlanan image_id bulundu; eşleşme belirsiz olabilir."
        )
        return result

    lookup = {
        row["_norm_image_id"]: row
        for _, row in metadata.iterrows()
        if row["_norm_image_id"]
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
            image_id = normalize_image_id(record["name"])
            row = lookup.get(image_id)

            if row is None:
                missing_files.append(record["name"])
                continue

            result["matched"] += 1
            actual_dx = row["_norm_dx"]
            lesion_id = row["_norm_lesion_id"]

            if actual_dx != expected_dx:
                mismatched_labels.append(
                    f"{record['name']} için metadata dx='{actual_dx}', "
                    f"beklenen '{expected_dx}'."
                )

            if not lesion_id:
                missing_lesions.append(record["name"])
            else:
                selected.append(
                    {
                        "group": group_name,
                        "lesion_id": lesion_id,
                        "file": record["name"],
                    }
                )

    if missing_files:
        result["messages"].append(
            f"Metadata içinde eşleşmeyen görsel sayısı: {len(missing_files)}."
        )

    if mismatched_labels:
        result["messages"].append(
            f"dx etiketi grup adıyla uyuşmayan görsel sayısı: {len(mismatched_labels)}."
        )

    if missing_lesions:
        result["messages"].append(
            f"lesion_id değeri boş olan görsel sayısı: {len(missing_lesions)}."
        )

    lesion_duplicates = []
    for group_name in [GROUP_NV, GROUP_MEL]:
        ids = [
            item["lesion_id"]
            for item in selected
            if item["group"] == group_name
        ]
        if len(ids) != len(set(ids)):
            lesion_duplicates.append(group_name)

    if lesion_duplicates:
        result["messages"].append(
            "Aynı grupta birden fazla görselin aynı lesion_id değerine sahip olduğu görüldü."
        )

    nv_lesions = {
        item["lesion_id"]
        for item in selected
        if item["group"] == GROUP_NV
    }
    mel_lesions = {
        item["lesion_id"]
        for item in selected
        if item["group"] == GROUP_MEL
    }

    if nv_lesions.intersection(mel_lesions):
        result["messages"].append(
            "İki grup arasında ortak lesion_id bulundu."
        )

    valid = (
        not missing_files
        and not mismatched_labels
        and not missing_lesions
        and not lesion_duplicates
        and not nv_lesions.intersection(mel_lesions)
    )

    result["valid"] = bool(valid)

    if valid:
        result["messages"].append(
            "Yüklenen görsellerin image_id/dx eşleşmesi ve gruplar içinde "
            "lesion_id tekilliği kontrol edildi."
        )

    result["messages"].append(
        "Metadata tek başına tohum 42 ile örnek seçiminin yapıldığını doğrulamaz; "
        "örnek seçim kaydı ayrıca saklanmalıdır."
    )
    return result


def batch_signature(nv_records, mel_records, metadata_bytes):
    parts = []
    for group_name, records in [
        ("nv", nv_records),
        ("mel", mel_records),
    ]:
        for record in records:
            parts.append(f"{group_name}:{record['normalized_name']}:{record['md5']}")

    if metadata_bytes is not None:
        parts.append("metadata:" + hashlib.md5(metadata_bytes).hexdigest())

    text = "|".join(sorted(parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analyze_batch_group(records, group_name):
    rows = []
    errors = []

    for record in records:
        try:
            image = image_from_bytes(record["bytes"])
            analysis = gorsel_analiz(
                image,
                k=2,
                n_angles=180,
                step_px=4,
                center_adjust=False,
                with_heat=False,
            )
            features = lesion_features(analysis["rgb"])

            rows.append(
                {
                    "grup": group_name,
                    "dosya": record["name"],
                    "A_U": analysis["U"],
                    "A_S": analysis["S"],
                    "B_p": analysis["B_chance"],
                    "p_acik_piksel": analysis["p"],
                    "kapsama": analysis["coverage"],
                    "B_edge": features["B_edge"],
                    "C_colors": features["C_colors"],
                    "renk_siniflari": features["color_names"],
                    "segmentasyon_gecerli": features["segmentation_valid"],
                    "lezyon_alan_orani": features["area_fraction"],
                }
            )
        except Exception as exc:
            errors.append(f"{record['name']}: {exc}")

    return rows, errors


def run_batch(nv_records, mel_records):
    nv_rows, nv_errors = analyze_batch_group(nv_records, GROUP_NV)
    mel_rows, mel_errors = analyze_batch_group(mel_records, GROUP_MEL)

    return {
        "data": pd.DataFrame(nv_rows + mel_rows),
        "errors": nv_errors + mel_errors,
        "nv_processed": len(nv_rows),
        "mel_processed": len(mel_rows),
    }


def build_batch_statistics(data):
    metric_specs = [
        {
            "metric": "A — Dönel simetri ölçümü (U)",
            "column": "A_U",
            "high_group": GROUP_NV,
            "low_group": GROUP_MEL,
            "high_chart": GROUP_NV_CHART,
            "low_chart": GROUP_MEL_CHART,
            "direction": "İyi huylu nevüs (nv) > kötü huylu melanom (mel)",
            "difference_label": "nv − mel",
        },
        {
            "metric": "B — Kenar düzensizliği indeksi",
            "column": "B_edge",
            "high_group": GROUP_MEL,
            "low_group": GROUP_NV,
            "high_chart": GROUP_MEL_CHART,
            "low_chart": GROUP_NV_CHART,
            "direction": "Kötü huylu melanom (mel) > iyi huylu nevüs (nv)",
            "difference_label": "mel − nv",
        },
        {
            "metric": "C — Renk sınıfı sayısı",
            "column": "C_colors",
            "high_group": GROUP_MEL,
            "low_group": GROUP_NV,
            "high_chart": GROUP_MEL_CHART,
            "low_chart": GROUP_NV_CHART,
            "direction": "Kötü huylu melanom (mel) > iyi huylu nevüs (nv)",
            "difference_label": "mel − nv",
        },
    ]

    results = []

    for spec in metric_specs:
        high_values = data.loc[
            data["grup"] == spec["high_group"],
            spec["column"],
        ]
        low_values = data.loc[
            data["grup"] == spec["low_group"],
            spec["column"],
        ]

        test = welch_directional(high_values, low_values)
        nv_mean = pd.to_numeric(
            data.loc[data["grup"] == GROUP_NV, spec["column"]],
            errors="coerce",
        ).mean()
        mel_mean = pd.to_numeric(
            data.loc[data["grup"] == GROUP_MEL, spec["column"]],
            errors="coerce",
        ).mean()

        comment = make_comment(
            spec["metric"],
            spec["high_chart"],
            spec["low_chart"],
            test,
        )

        significant = (
            test["n_high"] >= 2
            and test["n_low"] >= 2
            and test["difference"] > 0
            and test["p_one"] < ALFA_DUZELTILMIS
        )

        results.append(
            {
                **spec,
                **test,
                "nv_mean": float(nv_mean) if np.isfinite(nv_mean) else float("nan"),
                "mel_mean": float(mel_mean) if np.isfinite(mel_mean) else float("nan"),
                "comment": comment,
                "significant": bool(significant),
            }
        )

    return results


def results_table(results):
    rows = []

    for item in results:
        rows.append(
            {
                "Ölçüt": item["metric"],
                "Beklenen yön": item["direction"],
                "İyi huylu nevüs (nv) ortalaması": fmt_number(item["nv_mean"]),
                "Kötü huylu melanom (mel) ortalaması": fmt_number(item["mel_mean"]),
                f"Beklenen yönlü fark ({item['difference_label']})": fmt_number(item["difference"]),
                "Welch t": fmt_number(item["t"], 3),
                "Serbestlik derecesi": fmt_number(item["df"], 2),
                "Tek yönlü p": fmt_p(item["p_one"]),
                "İki yönlü p": fmt_p(item["p_two"]),
                "Cohen d": fmt_number(item["cohen_d"], 3),
                "Beklenen yönlü fark için %95 GA": (
                    f"[{fmt_number(item['ci_low'])}, {fmt_number(item['ci_high'])}]"
                    if np.isfinite(item["ci_low"]) and np.isfinite(item["ci_high"])
                    else "—"
                ),
                "Düzeltilmiş eşik": f"{ALFA_DUZELTILMIS:.4f}",
                "Geçerli n (yüksek/düşük)": f"{item['n_high']}/{item['n_low']}",
            }
        )

    return pd.DataFrame(rows)


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
    Ayna/eksen yansıma simetrisi ölçümü değildir.
    """
)
st.sidebar.caption(
    f"{SURUM} · {n_angles} açı örneği · {step_px} px radyal adım"
)


# ---------------------------------------------------------------------
# SAYFA BAŞLIĞI
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

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
            "Girih esintili şematik on katlı motiftir; tarihî bir desenin birebir örneği değildir."
        )

    st.caption(
        "Kuasikristal keşfi Dan Shechtman ile ilişkilidir; yukarıdaki çizimlerden "
        "hiçbiri onun deneysel görüntüsü değildir."
    )


# ---------------------------------------------------------------------
# 2. KALİBRASYON VE SENTETİK ÖRNEK
# ---------------------------------------------------------------------

with tabs[1]:
    st.markdown("### Sentetik örnek ölçümü")

    st.info(
        "Bu bölüm seçilen tek bir sentetik örneğin ölçümünü gösterir. "
        "Tek başına bütün doğrulama testlerinin tamamlandığı anlamına gelmez."
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
        st.image(sample_result["rgb"], caption="Sentetik örnek", use_container_width=True)
    with binary_col:
        st.image(
            sample_result["binary"] * 255,
            caption="Ortalama parlaklık eşiğiyle ikileştirme",
            use_container_width=True,
        )
    with heat_col:
        st.pyplot(
            fig_heat_overlay(
                sample_result["rgb"],
                sample_result["heatmap"],
                "Sentetik örneğin yerel U haritası",
            ),
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div class="section-card">
        Ölçüm ayarları: <b>k={selected_k}</b>,
        <b>{n_angles} açı örneği</b>, <b>{step_px} px radyal adım</b>.
        Analizde bu örneğin kendi açık piksel oranından hesaplanan B(p) kullanıldı.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Isı haritasındaki renkler yerel U değerlerini görselleştirir; "
        "tek başına bir doğrulama sonucu veya kusur kararı değildir."
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

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        uploaded_a = st.file_uploader(
            "1. görsel (isteğe bağlı)",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_image_a",
        )
    with input_col2:
        uploaded_b = st.file_uploader(
            "2. görsel — karşılaştırma için (isteğe bağlı)",
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
                st.markdown(f"**Görsel 1: {source_a}**")
                st.image(result_a["rgb"], use_container_width=True)
                st.metric("Görsel 1 — Uₖ", fmt_number(result_a["U"]))
                st.caption(
                    f"Sₖ={fmt_number(result_a['S'])} · "
                    f"B(p)={fmt_number(result_a['B_chance'])} · "
                    f"kapsama={result_a['coverage']:.1%}"
                )
                st.pyplot(
                    fig_heat_overlay(
                        result_a["rgb"],
                        result_a["heatmap"],
                        "Görsel 1 — yerel U haritası",
                    ),
                    use_container_width=True,
                )

            with col_b:
                st.markdown(f"**Görsel 2: {source_b}**")
                st.image(result_b["rgb"], use_container_width=True)
                st.metric("Görsel 2 — Uₖ", fmt_number(result_b["U"]))
                st.caption(
                    f"Sₖ={fmt_number(result_b['S'])} · "
                    f"B(p)={fmt_number(result_b['B_chance'])} · "
                    f"kapsama={result_b['coverage']:.1%}"
                )
                st.pyplot(
                    fig_heat_overlay(
                        result_b["rgb"],
                        result_b["heatmap"],
                        "Görsel 2 — yerel U haritası",
                    ),
                    use_container_width=True,
                )

            comparison = pd.DataFrame(
                [
                    {
                        "Ölçüt": "Sₖ — eşleşme oranı",
                        "Görsel 1": result_a["S"],
                        "Görsel 2": result_b["S"],
                        "Fark (Görsel 2 − Görsel 1)": result_b["S"] - result_a["S"],
                    },
                    {
                        "Ölçüt": "B(p) — görüntüye ait şans düzeyi",
                        "Görsel 1": result_a["B_chance"],
                        "Görsel 2": result_b["B_chance"],
                        "Fark (Görsel 2 − Görsel 1)": (
                            result_b["B_chance"] - result_a["B_chance"]
                        ),
                    },
                    {
                        "Ölçüt": "Uₖ — normalize dönel ölçüm",
                        "Görsel 1": result_a["U"],
                        "Görsel 2": result_b["U"],
                        "Fark (Görsel 2 − Görsel 1)": result_b["U"] - result_a["U"],
                    },
                ]
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)

        else:
            st.markdown(f"#### İncelenen görsel: {source_a}")
            image_col, binary_col, heat_col = st.columns(3)

            with image_col:
                st.image(result_a["rgb"], caption="200×200 RGB görüntü", use_container_width=True)
            with binary_col:
                st.image(
                    result_a["binary"] * 255,
                    caption="İkileştirilmiş görüntü",
                    use_container_width=True,
                )
            with heat_col:
                st.pyplot(
                    fig_heat_overlay(
                        result_a["rgb"],
                        result_a["heatmap"],
                        "Yerel U haritası",
                    ),
                    use_container_width=True,
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Eşleşme oranı Sₖ", fmt_number(result_a["S"]))
            m2.metric("Görüntüye ait B(p)", fmt_number(result_a["B_chance"]))
            m3.metric("Normalize ölçüm Uₖ", fmt_number(result_a["U"]))
            m4.metric("Geçerli çift kapsaması", f"{result_a['coverage']:.1%}")

        st.caption(
            "Isı haritası yerel eşleşme örüntüsünü gösterir. Renkler tek başına "
            "bir nesne, lezyon veya kusur hakkında karar vermez."
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

    use_uploaded_pair = reference_upload is not None and prototype_upload is not None

    if use_uploaded_pair:
        try:
            reference_image = image_from_bytes(reference_upload.getvalue())
            prototype_image = image_from_bytes(prototype_upload.getvalue())
            reference_name = reference_upload.name
            prototype_name = prototype_upload.name
        except Exception as exc:
            st.error(f"Yüklenen görseller açılamadı: {exc}")
            reference_image = sentetik_gorsel("fivefold", seed=42)
            prototype_image = sentetik_gorsel("fivefold", disturbance=disturbance, seed=42)
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
        st.markdown(f"**Referans: {reference_name}**")
        st.image(reference_result["rgb"], use_container_width=True)
        st.metric("Referans Uₖ", fmt_number(reference_result["U"]))
        st.caption(
            f"Sₖ={fmt_number(reference_result['S'])} · "
            f"B(p)={fmt_number(reference_result['B_chance'])}"
        )

    with industry_col2:
        st.markdown(f"**Prototip: {prototype_name}**")
        st.image(prototype_result["rgb"], use_container_width=True)
        st.metric("Prototip Uₖ", fmt_number(prototype_result["U"]))
        st.caption(
            f"Sₖ={fmt_number(prototype_result['S'])} · "
            f"B(p)={fmt_number(prototype_result['B_chance'])}"
        )

    delta_u = prototype_result["U"] - reference_result["U"]
    st.metric("Ölçülen fark (prototip Uₖ − referans Uₖ)", fmt_number(delta_u))

    heat_col, note_col = st.columns([1.1, 1])
    with heat_col:
        st.pyplot(
            fig_heat_overlay(
                prototype_result["rgb"],
                prototype_result["heatmap"],
                "Prototip örneğinin yerel U haritası",
            ),
            use_container_width=True,
        )
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
            "Varsayılan sentetik örnek beş katlıdır; bu nedenle başlangıç mertebesi k=5'tir."
        )


# ---------------------------------------------------------------------
# 5. HAM10000 GRUP KARŞILAŞTIRMASI
# ---------------------------------------------------------------------

with tabs[4]:
    st.markdown("### İyi huylu nevüs ve kötü huylu melanom etiketli görseller")

    st.info(
        f"**{GROUP_NV}** ile **{GROUP_MEL}** ayrı gruplar olarak karşılaştırılır. "
        "Bu adlandırmalar HAM10000 veri setindeki dx etiketlerini ifade eder."
    )

    st.markdown(
        f"""
        <div class="section-card">
        <b>Ana karşılaştırma ayarları sabittir:</b>
        100 nv + 100 mel görsel, k=2 (180° dönel karşılaştırma),
        Hassas örnekleme (180 açı, 4 px adım), merkez ayarı kapalı ve
        her görüntü için ayrı B(p). Üç ölçüt için Bonferroni eşiği
        α={ALFA_DUZELTILMIS:.4f}'tür.
        <br><br>
        Uygulama yüklenen dosyaları analiz eder; 100/100 dışındaki yüklemeler
        ana sonuç değil, ön inceleme olarak değerlendirilir.
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
        "İsteğe bağlı: HAM10000 metadata CSV (image_id, lesion_id, dx sütunları)",
        type=["csv"],
        key="batch_metadata",
    )

    nv_records = uploaded_records(nv_files)
    mel_records = uploaded_records(mel_files)
    metadata_bytes = metadata_upload.getvalue() if metadata_upload is not None else None

    duplicate_issues = duplicate_report(
        {
            GROUP_NV: nv_records,
            GROUP_MEL: mel_records,
        }
    )

    meta_check = check_metadata(
        metadata_bytes,
        nv_records,
        mel_records,
    )

    count_col1, count_col2, count_col3 = st.columns(3)
    count_col1.metric("İyi huylu nevüs (nv) dosyası", len(nv_records))
    count_col2.metric("Kötü huylu melanom (mel) dosyası", len(mel_records))
    count_col3.metric(
        "100 / 100 görüntü sayısı",
        "Sağlandı" if len(nv_records) == 100 and len(mel_records) == 100 else "Sağlanmadı",
    )

    if duplicate_issues:
        st.error(
            "Tekrarlanan içerik veya dosya adı bulundu. Tekrarları temizleyip "
            "yeniden yükleyin; bu durumda analiz başlatılmaz."
        )
        with st.expander("Tekrar denetimi ayrıntıları"):
            for issue in duplicate_issues:
                st.write("• " + issue)

    if metadata_bytes is None:
        st.warning(meta_check["messages"][0])
    else:
        if meta_check["valid"]:
            st.success(
                "Metadata ile image_id/dx eşleşmesi ve lesion_id tekilliği kontrol edildi."
            )
        else:
            st.warning("Metadata denetimi tam doğrulama vermedi.")
        with st.expander("Metadata denetimi ayrıntıları"):
            for message in meta_check["messages"]:
                st.write("• " + message)

    st.caption(
        "Metadata denetimi, tohum 42 ile örnek seçiminin yapıldığını tek başına doğrulamaz; "
        "seçim kaydı ayrıca saklanmalıdır."
    )

    can_run = (
        len(nv_records) >= 2
        and len(mel_records) >= 2
        and not duplicate_issues
    )

    signature = batch_signature(nv_records, mel_records, metadata_bytes)

    run_button = st.button(
        "HAM10000 grup analizini çalıştır",
        disabled=not can_run,
        key="run_batch",
    )

    if not can_run:
        st.caption(
            "Analiz için her grupta en az iki görsel yükleyin ve tekrarları giderin."
        )

    if run_button:
        with st.spinner("Görseller işleniyor; sabit ana karşılaştırma ayarları uygulanıyor..."):
            batch_result = run_batch(nv_records, mel_records)
            batch_result["signature"] = signature
            batch_result["metadata_valid"] = meta_check["valid"]
            batch_result["uploaded_nv"] = len(nv_records)
            batch_result["uploaded_mel"] = len(mel_records)
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
            processed_nv = int(saved_batch["nv_processed"])
            processed_mel = int(saved_batch["mel_processed"])

            count_ok = processed_nv == 100 and processed_mel == 100
            no_errors = len(errors) == 0
            metadata_ok = bool(saved_batch.get("metadata_valid", False))

            st.markdown("#### Yükleme ve protokol durumu")

            if count_ok and no_errors and metadata_ok:
                st.success(
                    "100 iyi huylu nevüs (nv) ve 100 kötü huylu melanom (mel) görseli "
                    "işlendi; metadata etiketleri ve lesion_id tekilliği kontrol edildi."
                )
                st.caption(
                    "Tohum 42 ile örnek seçiminin kaydı bu uygulama tarafından "
                    "bağımsız olarak doğrulanmaz."
                )
            elif count_ok and no_errors:
                st.warning(
                    "100/100 görüntü işlendi; ancak lesion_id tekilliği metadata ile "
                    "doğrulanmadığı için bu çıktı tam ana protokol doğrulaması olarak "
                    "işaretlenemez."
                )
            else:
                st.warning(
                    "Bu yükleme 100/100 ana protokol koşulunu karşılamıyor veya bazı "
                    "dosyalar işlenemedi. Aşağıdaki sonuçlar yalnızca ön incelemedir."
                )

            processed_col1, processed_col2 = st.columns(2)
            processed_col1.metric("İşlenen nv görseli", processed_nv)
            processed_col2.metric("İşlenen mel görseli", processed_mel)

            if errors:
                with st.expander(f"Okunamayan dosyalar ({len(errors)})"):
                    for error in errors:
                        st.write("• " + error)

            st.markdown("#### Ölçütlerin grup dağılımı")
            st.pyplot(
                fig_batch_boxplots(batch_data),
                use_container_width=True,
            )

            st.markdown(
                """
                **A, B ve C ayrı ölçütlerdir.** Birleşik puan oluşturulmaz.
                `B(p)` şans düzeyi ile `B` kenar düzensizliği ölçütü farklı şeylerdir.
                """
            )

            batch_stats = build_batch_statistics(batch_data)
            stats_df = results_table(batch_stats)
            st.markdown("#### İstatistiksel karşılaştırma")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            for item in batch_stats:
                st.markdown(
                    f"""
                    <div class="result-card">
                    <b>{item['metric']}</b><br>
                    {item['comment']}<br>
                    <span class="small-muted">
                    Beklenen yönlü fark: {fmt_number(item['difference'])} ·
                    Tek yönlü p: {fmt_p(item['p_one'])} ·
                    İki yönlü p: {fmt_p(item['p_two'])} ·
                    Cohen d: {fmt_number(item['cohen_d'], 3)}
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Anlamlılık değerlendirmesi, önceden belirlenen yönlü Welch testi ve "
                f"üç ölçüt için Bonferroni eşiği α={ALFA_DUZELTILMIS:.4f} kullanılarak "
                "yapılır. Anlamlı fark bulunmaması, gruplar arasında hiç fark olmadığı "
                "anlamına gelmez."
            )

            display_columns = [
                "grup",
                "dosya",
                "A_U",
                "A_S",
                "B_p",
                "B_edge",
                "C_colors",
                "renk_siniflari",
                "kapsama",
                "segmentasyon_gecerli",
            ]

            rename_columns = {
                "grup": "Grup etiketi",
                "dosya": "Dosya",
                "A_U": "A — Dönel simetri U",
                "A_S": "A — Eşleşme oranı S",
                "B_p": "A — Görüntüye ait B(p)",
                "B_edge": "B — Kenar düzensizliği",
                "C_colors": "C — Renk sınıfı sayısı",
                "renk_siniflari": "Bulunan renk sınıfları",
                "kapsama": "Geçerli çift kapsaması",
                "segmentasyon_gecerli": "B/C segmentasyonu geçerli",
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
                f"100/100 boyut koşulu: {'Evet' if count_ok else 'Hayır'}",
                f"Metadata lesion_id kontrolü: {'Geçti' if metadata_ok else 'Doğrulanmadı'}",
                "Ölçüm ayarları: k=2, 180 açı, 4 px adım, merkez ayarı kapalı.",
                "B(p) her görüntü için ayrı hesaplanmıştır.",
                f"Üç ölçüt için Bonferroni eşiği: {ALFA_DUZELTILMIS:.4f}",
                "",
                "Ölçüm yorumları:",
            ]

            for item in batch_stats:
                txt_lines.append(f"- {item['metric']}: {item['comment']}")

            txt_lines.extend(
                [
                    "",
                    "Not: Fark bulunmaması gruplar arasında hiç fark olmadığı anlamına gelmez.",
                    "Seçim tohumu 42 ve örnek seçimi uygulama tarafından bağımsız doğrulanmaz.",
                    f"Uygulama sürümü: {SURUM}",
                    f"Ayarlar: k={k}, açı={n_angles}, radyal adım={step_px}px, merkez ayarı={center_adjust}",
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
        "Bu uygulama araştırma/prototip amaçlıdır. İstatistiksel grup farkları "
        "tek başına model doğrulaması, klinik performans veya tanı başarısı göstermez."
    )

    st.markdown(
        """
        <div class="section-card">
        YSÜM, görüntülerden dönel benzerlik ölçümü üretmek ve sonuçları
        ayrı görsel ölçütlerle birlikte incelemek için hazırlanmış bir
        proje prototipidir. Penrose/altın oran bağlantısı kuramsal bir
        referanstır; A, B ve C ölçümlerinin tümü bu matematikten türetilmiş
        değildir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### A — Dönel simetri ölçümü")
    st.markdown(
        """
        1. Görsel RGB biçimine çevrilip **200×200** boyutuna getirilir.
        2. Gri tonlama yapılır ve ortalama parlaklık eşiğiyle ikili görüntü oluşturulur.
        3. Merkez çevresinde seçilen açılarda ve yarıçaplarda, aralarında `360°/k`
           bulunan piksel çiftleri karşılaştırılır.
        4. Her açısal örnekte geçerli çiftlerin kapsaması en az **0,50** olmalıdır.
           Örneklenen çiftlerde görüntü değeri tamamen tek tip olan açısal örnekler
           ölçüme katılmaz.
        5. Eşleşme oranı `Sₖ`, o görüntünün açık piksel oranından hesaplanan
           `B(p)=p²+(1−p)²` ve normalize ölçüm `Uₖ` raporlanır.
        """
    )

    st.latex(r"U_k=\frac{S_k-B(p)}{1-B(p)}")

    st.markdown("#### Dönel mertebe neyi ifade eder?")
    st.markdown(
        """
        - **k=2:** 180° dönel karşılaştırma; ayna yansıması/eksen simetrisi değildir.
        - **k=5:** 72° aralıklı dönel karşılaştırma; Penrose döşemesi veya kuasikristal
          olduğu sonucunu tek başına vermez.
        - **k=6 ve k=8:** sırasıyla 60° ve 45° dönel karşılaştırmadır; belirli bir
          kristal, makine parçası ya da mimari geleneği kanıtlamaz.
        """
    )

    st.markdown("#### B ve C — Ek görüntü ölçütleri")
    st.markdown(
        """
        - **B — Kenar düzensizliği:** Otsu bölütlemesiyle seçilen merkez bileşeninin
          sınırından hesaplanan basit çevre²/(4π×alan) indeksidir. Büyük değer,
          bu ölçüme göre daha uzun/düzensiz bir sınırla uyumludur.
        - **C — Renk sınıfı sayısı:** sabit altı renk paletindeki sınıflardan,
          bölütlenen alanın en az %5'ini oluşturanların sayısıdır.
        - B ve C, Penrose matematiğinden türetilmiş ölçüler değildir.
          A, B ve C birleştirilerek doğrulanmamış tek bir puan oluşturulmaz.
        - `B(p)` A ölçümündeki şans eşleşme düzeyidir; `B` kenar düzensizliği
          ölçütüyle karıştırılmamalıdır.
        """
    )

    st.markdown("#### Isı haritası")
    st.markdown(
        """
        Isı haritası, ikili görüntü ile k katlı dönel karşılaştırma arasındaki
        yerel eşleşme oranını aynı görüntüye ait `B(p)` tabanına göre görselleştirir.
        Isı haritası yerel gösterimdir; tek başına kusur, tanı veya kabul/red kararı
        üretmez.
        """
    )

    st.markdown("#### Kapsam ve sınırlılıklar")
    st.markdown(
        """
        - Sonuçlar kullanılan görüntü kalitesine, boyutlandırmaya, eşiklemeye,
          merkez seçimine ve örnekleme ayarlarına bağlıdır.
        - Sentetik motifler gerçek Penrose döşemesi veya deneysel kırınım fotoğrafı değildir.
        - HAM10000 karşılaştırması, veri setinde etiketlenmiş görüntü gruplarının
          ölçümlerini karşılaştırır; uygulama tek bir görsel için klinik tanı üretmez.
        - Tek görüntü analiziyle üretim kusuru tespiti veya üretime uygunluk kararı verilmez.
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
        "HAM10000 veri seti için veri kümesi kaynağını, sürümünü ve indirme tarihini "
        "raporda belirtin; veri kümesinin lisans koşullarına uyun."
    )
