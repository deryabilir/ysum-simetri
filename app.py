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

SURUM = "YSÜM-v4.1-AE"
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
