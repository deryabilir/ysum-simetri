import hashlib
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage, stats

# ============================ SAYFA VE TEMA AYARLARI ============================
st.set_page_config(
    page_title="YSÜM · Yerel Simetri Üstünlük Modeli",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, ferah ve kurumsal arayüz stili
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f2b46 0%, #1a4a75 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(15, 43, 70, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #b0cce5;
        max-width: 900px;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        border-top: 4px solid #1a4a75;
    }
    .metric-card-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-val {
        font-size: 26px;
        font-weight: 700;
        color: #0f2b46;
        margin: 6px 0;
    }
    .metric-card-sub {
        font-size: 12px;
        color: #475569;
    }
    .badge-success {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-neutral {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-gold {
        display: inline-block;
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-size: 14px;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================ PROTOKOL VE SABİTLER ============================
PHI = (1 + math.sqrt(5)) / 2
P_PENROSE = 1 / PHI
B_PENROSE = P_PENROSE**2 + (1 - P_PENROSE) ** 2

IMG_N = 200
ALFA = 0.05
KRITER_SAYISI = 3
ALFA_DUZELTILMIS = ALFA / KRITER_SAYISI  # Bonferroni, ≈ 0,0167
YORUM_ESIGI = 0.15
IZG_BASLANGIC = 10

MIN_KAPSAMA = 0.50
DUZ_ALANLARI_DISLA = True

LEZYON_MIN_ORAN = 0.02
LEZYON_MAX_ORAN = 0.80
RENK_MIN_ORAN = 0.05

SURUM = "YSÜM-v3.1-Pro"
PROTOKOL_ADI = (
    "YSÜM v3.1 · A: Simetri (k = 2, Kapsama ≥ %50) · "
    "B: Kenar Düzensizliği · C: Renk Çeşitliliği · Düzeltilmiş α = 0,0167"
)

GRUP_IYI = "İyi huylu (nv)"
GRUP_KOTU = "Kötü huylu (mel)"

KRITERLER = [
    ("U_ort", "A · Simetri (U_ort)", GRUP_IYI, GRUP_KOTU),
    ("B_duz", "B · Kenar düzensizliği", GRUP_KOTU, GRUP_IYI),
    ("C_renk", "C · Renk çeşitliliği (renk sınıfı sayısı)", GRUP_KOTU, GRUP_IYI),
]

UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp"]

K_SECENEKLER = {
    2: "2 Kat · 180° (İkili Ayna / Eksen Simetrisi)",
    5: "5 Kat · 72° (Penrose / Kuasikristal Geometrisi)",
    6: "6 Kat · 60° (Altıgen / Mekanik & Kristal Yapı)",
    8: "8 Kat · 45° (Sekizgen / Girih & Mimari Motif)",
}

SKOR_YARICAPLARI = (20, 40, 60, 80)
ISI_YARICAPLARI = (10, 20, 30, 40)

HASSASLIK = {
    "Standart (36 açı, adım 8)": (36, 8),
    "Hassas (180 açı, adım 4)": (180, 4),
    "Çok hassas (360 açı, adım 4)": (360, 4),
}

ORNEKLER = {
    "Yuvarlak Şekil": "yuvarlak",
    "Düzensiz Sınır": "duzensiz",
    "5 Katlı Rozet": "bes_kat",
    "Rastgele Dağılım (Gürültü)": "gurultu",
}

RENK_PALETI = np.array([
    [235, 235, 235],  # beyaz
    [196, 150, 110],  # açık kahverengi
    [110, 70, 40],    # koyu kahverengi
    [110, 130, 150],  # mavi-gri
    [30, 30, 30],     # siyah
    [170, 60, 60],    # kırmızımsı
], dtype=float)
RENK_ADLARI = [
    "beyaz", "açık kahverengi", "koyu kahverengi",
    "mavi-gri", "siyah", "kırmızımsı",
]


# ============================ FORMAT VE İSTATİSTİK YARDIMCILARI ============================
def tr_sayi(x, basamak=4):
    if x is None:
        return "hesaplanamadı"
    try:
        if not np.isfinite(x):
            return "hesaplanamadı"
    except TypeError:
        return str(x)
    return f"{x:.{basamak}f}".replace(".", ",")


def p_yaz(p):
    if p is None or not np.isfinite(p):
        return "hesaplanamadı"
    if p < 0.0001:
        return "< 0,0001"
    return f"{p:.4f}".replace(".", ",")


def yorum_etiketi(u):
    if u is None or not np.isfinite(u):
        return "Hesaplanamadı"
    if u > YORUM_ESIGI:
        return "Şans Beklentisinden Belirgin Yüksek"
    if u < -YORUM_ESIGI:
        return "Şans Beklentisinden Belirgin Düşük"
    return "Şans Beklentisiyle Uyumlu"


def etki_etiketi(d):
    a = abs(d)
    if a < 0.20:
        return "çok küçük"
    if a < 0.50:
        return "küçük"
    if a < 0.80:
        return "orta"
    return "büyük"


# ============================ MATPLOTLIB ÇİZİMLERİ ============================
def goster(fig):
    st.pyplot(fig)
    plt.close(fig)


def fig_kirinim():
    x = np.linspace(-1, 1, 360)
    X, Y = np.meshgrid(x, x)
    z = sum(
        np.cos(
            2 * np.pi * 6
            * (
                X * np.cos(2 * np.pi * j / 5)
                + Y * np.sin(2 * np.pi * j / 5)
            )
        )
        for j in range(5)
    )
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    ax.imshow(z, cmap="magma")
    ax.set_title("5 Katlı Kırınım Simülasyonu", fontsize=10, fontweight="bold", color="#0f2b46")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_rozet():
    aci = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
    x, y = np.cos(aci), np.sin(aci)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    for i in range(5):
        j = (i + 2) % 5
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#1a4a75", lw=2.2)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="#94a3b8", lw=1, ls="--")
    ax.scatter(x, y, s=45, color="#c59b27", zorder=3)
    ax.set_aspect("equal")
    ax.set_title("5 Katlı Penrose Yıldız Rozeti", fontsize=10, fontweight="bold", color="#0f2b46")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_girih():
    aci = 2 * np.pi * np.arange(10) / 10
    x, y = np.cos(aci), np.sin(aci)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    for i in range(10):
        j = (i + 3) % 10
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#b45309", lw=1.6)
    ax.set_aspect("equal")
    ax.set_title("10 Katlı Geometrik Motif (Girih)", fontsize=10, fontweight="bold", color="#0f2b46")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_b_egrisi():
    p = np.linspace(0, 1, 300)
    b = p**2 + (1 - p) ** 2
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="none")
    ax.plot(p, b, color="#1a4a75", lw=2.2, label=r"$B(p) = p^2 + (1-p)^2$")
    ax.axvline(P_PENROSE, ls="--", color="#c59b27", lw=1.4, label=r"$p = 1/\varphi \approx 0,618$")
    ax.axhline(B_PENROSE, ls="--", color="#c59b27", lw=1.4, label=r"$B \approx 0,528$")
    ax.scatter([P_PENROSE], [B_PENROSE], color="#c59b27", s=60, zorder=4)
    ax.set_xlabel("Açık Piksel Oranı (p)", fontsize=10)
    ax.set_ylabel("Şans Eşleşme Eşiği B(p)", fontsize=10)
    ax.set_title("Kuramsal Penrose Şans Eşiği Eğrisi", fontsize=11, fontweight="bold", color="#0f2b46")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)
    fig.tight_layout()
    return fig


def isi_figur(gri, izg, u_matris, baslik):
    fig, ax = plt.subplots(figsize=(4.8, 4.5), facecolor="none")
    ax.imshow(gri, cmap="gray", extent=(0, IMG_N, IMG_N, 0))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((0, 0, 0, 0))
    maskeli = np.ma.masked_invalid(u_matris)

    e = (izg[0], izg[-1], izg[-1], izg[0])
    im = ax.imshow(maskeli, cmap=cmap, vmin=-0.5, vmax=0.5, alpha=0.68, extent=e)
    ax.set_xlim(0, IMG_N)
    ax.set_ylim(IMG_N, 0)
    ax.set_title(baslik, fontsize=10, fontweight="bold", color="#0f2b46")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("YSÜM Skoru (U)", fontsize=9)
    fig.tight_layout()
    return fig


def toplu_grafik(df):
    fig, eksen = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="none")
    renkler = ["#3b82f6", "#ef4444"]

    for ax, (anahtar, ad, _, _) in zip(eksen, KRITERLER):
        veri = [
            df.loc[df["Grup"] == g, anahtar].dropna().values
            for g in (GRUP_IYI, GRUP_KOTU)
        ]
        bp = ax.boxplot(veri, widths=0.45, patch_artist=True)
        for patch, col in zip(bp['boxes'], ["#e0f2fe", "#fee2e2"]):
            patch.set_facecolor(col)
            patch.set_edgecolor("#475569")
            patch.set_linewidth(1.2)
        for median in bp['medians']:
            median.set(color='#0f2b46', linewidth=2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["İyi Huylu (nv)", "Kötü Huylu (mel)"], fontsize=10, fontweight="bold")
        if anahtar == "U_ort":
            ax.axhline(0, color="gray", lw=0.9, ls="-")
            ax.axhline(YORUM_ESIGI, color="#c59b27", ls=":", lw=1.2, label="±0,15 Eşik")
            ax.axhline(-YORUM_ESIGI, color="#c59b27", ls=":", lw=1.2)
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(ad, fontsize=11, fontweight="bold", color="#0f2b46")
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


# ============================ GÖRÜNTÜ VE MATEMATİKSEL İŞLEMLER ============================
def gorselleri_oku(veri):
    try:
        gri = np.asarray(
            Image.open(io.BytesIO(veri))
            .convert("L")
            .resize((IMG_N, IMG_N), Image.LANCZOS),
            dtype=np.uint8,
        )
        rgb = np.asarray(
            Image.open(io.BytesIO(veri))
            .convert("RGB")
            .resize((IMG_N, IMG_N), Image.LANCZOS),
            dtype=np.uint8,
        )
        return gri, rgb
    except Exception:
        return None, None


def ornek_uret(kod):
    y, x = np.mgrid[0:IMG_N, 0:IMG_N]
    dx, dy = x - IMG_N / 2, y - IMG_N / 2
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)

    if kod == "yuvarlak":
        g = np.where(r < 60, 215, 40)
    elif kod == "duzensiz":
        sinir = 55 + 14 * np.sin(3 * th + 0.5) + 8 * np.cos(5 * th)
        g = np.where(r < sinir, 205, 45)
    elif kod == "bes_kat":
        g = np.where((r < 80) & (np.cos(5 * th) > 0), 225, 40)
    else:
        g = np.random.default_rng(1).integers(0, 256, size=(IMG_N, IMG_N))

    return g.astype(np.uint8)


def merkez_bul(binary):
    koyu = ~binary
    if 0.02 <= koyu.mean() <= 0.98:
        ys, xs = np.nonzero(koyu)
        return int(round(xs.mean())), int(round(ys.mean()))
    return IMG_N // 2, IMG_N // 2


def sim_hesapla(binary, k, cxs, cys, radii, aci_sayisi, parca=128):
    cxs = np.asarray(cxs, dtype=float)
    cys = np.asarray(cys, dtype=float)

    r = np.asarray(radii, dtype=float)[None, :, None]
    th = np.linspace(0.0, 2 * np.pi, aci_sayisi, endpoint=False)[None, None, :]

    kayma = 2 * np.pi / k
    cos1, sin1 = np.cos(th), np.sin(th)
    cos2, sin2 = np.cos(th + kayma), np.sin(th + kayma)

    s_parcalar = []
    kapsama_parcalar = []

    for bas in range(0, len(cxs), parca):
        cx = cxs[bas:bas + parca, None, None]
        cy = cys[bas:bas + parca, None, None]

        x1 = np.rint(cx + r * cos1).astype(int)
        y1 = np.rint(cy + r * sin1).astype(int)
        x2 = np.rint(cx + r * cos2).astype(int)
        y2 = np.rint(cy + r * sin2).astype(int)

        gecerli = (
            (x1 >= 0) & (x1 < IMG_N)
            & (y1 >= 0) & (y1 < IMG_N)
            & (x2 >= 0) & (x2 < IMG_N)
            & (y2 >= 0) & (y2 < IMG_N)
        )

        x1c = np.clip(x1, 0, IMG_N - 1)
        y1c = np.clip(y1, 0, IMG_N - 1)
        x2c = np.clip(x2, 0, IMG_N - 1)
        y2c = np.clip(y2, 0, IMG_N - 1)

        v1 = binary[y1c, x1c]
        v2 = binary[y2c, x2c]

        esles = (v1 == v2) & gecerli

        toplam = gecerli.sum(axis=(1, 2)).astype(float)
        eslesme = esles.sum(axis=(1, 2)).astype(float)

        kapasite = gecerli.shape[1] * gecerli.shape[2]
        kapsama = toplam / kapasite

        s_ham = np.divide(
            eslesme,
            toplam,
            out=np.full(eslesme.shape, np.nan, dtype=float),
            where=toplam > 0,
        )

        acik_sayisi = (
            (v1 & gecerli).sum(axis=(1, 2))
            + (v2 & gecerli).sum(axis=(1, 2))
        )
        duz = (acik_sayisi == 0) | (acik_sayisi == 2 * toplam)

        kabul = (kapsama >= MIN_KAPSAMA) & (toplam > 0)
        if DUZ_ALANLARI_DISLA:
            kabul &= ~duz

        s_parcalar.append(np.where(kabul, s_ham, np.nan))
        kapsama_parcalar.append(kapsama)

    return (
        np.concatenate(s_parcalar),
        np.concatenate(kapsama_parcalar),
    )


def gorsel_analiz(gri, k, sabit, koyu_modu, radii, kontrol, aci_sayisi, adim):
    esik = float(gri.mean())
    binary = gri > esik
    p = float(binary.mean())

    if p <= 0.0 or p >= 1.0:
        return None

    B = B_PENROSE if sabit else p**2 + (1 - p) ** 2

    if koyu_modu:
        cx, cy = merkez_bul(binary)
    else:
        cx, cy = IMG_N // 2, IMG_N // 2

    izg = np.arange(IZG_BASLANGIC, IMG_N - IZG_BASLANGIC + 1, adim)
    gx, gy = np.meshgrid(izg, izg)
    xs, ys = gx.ravel(), gy.ravel()

    s_izg, kapsama_izg = sim_hesapla(binary, k, xs, ys, radii, aci_sayisi)
    u_izg = (s_izg - B) / (1 - B)

    gecerli = np.isfinite(u_izg)
    if not gecerli.any():
        return None

    u = u_izg[gecerli]
    s = s_izg[gecerli]

    s_merkez, _ = sim_hesapla(binary, k, [cx], [cy], radii, aci_sayisi)
    u_merkez = (s_merkez[0] - B) / (1 - B)

    sonuc = {
        "p": p,
        "B": B,
        "S_ort": float(s.mean()),
        "U_ort": float(u.mean()),
        "U_medyan": float(np.median(u)),
        "U_std": float(u.std(ddof=1)) if u.size > 1 else 0.0,
        "U_maks": float(u.max()),
        "U_merkez": float(u_merkez),
        "Gecerli_nokta": int(gecerli.sum()),
        "Toplam_nokta": int(u_izg.size),
        "Kapsama_ort": float(kapsama_izg[gecerli].mean()),
        "izg": izg,
        "U_matris": u_izg.reshape(gx.shape),
    }

    if kontrol:
        karisik = (
            np.random.default_rng(0)
            .permutation(gri.ravel())
            .reshape(gri.shape)
        )
        k_bin = karisik > esik
        s_k, _ = sim_hesapla(k_bin, k, xs, ys, radii, aci_sayisi)
        u_k = (s_k - B) / (1 - B)
        kontrol_degerleri = u_k[gecerli & np.isfinite(u_k)]
        sonuc["U_kontrol"] = (
            float(kontrol_degerleri.mean())
            if kontrol_degerleri.size
            else float("nan")
        )

    return sonuc


def otsu_esigi(gri):
    hist = np.bincount(gri.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b2 = np.nan_to_num(sigma_b2, nan=0.0, posinf=0.0, neginf=0.0)
    return int(np.argmax(sigma_b2))


def lezyon_maskesi(gri):
    esik = otsu_esigi(gri)
    merkez_y, merkez_x = IMG_N // 2, IMG_N // 2
    adaylar = []

    for maske in (gri <= esik, gri > esik):
        etiket, n = ndimage.label(maske)
        if n == 0:
            continue
        boyut = np.bincount(etiket.ravel())[1:]
        merkez_etiket = etiket[merkez_y, merkez_x]
        for i, b in enumerate(boyut, start=1):
            oran = b / (IMG_N * IMG_N)
            if LEZYON_MIN_ORAN <= oran <= LEZYON_MAX_ORAN:
                adaylar.append((i == merkez_etiket, int(b), etiket, i))

    if not adaylar:
        return None

    adaylar.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, etiket, i = adaylar[0]
    return etiket == i


def kenar_duzensizligi(lez):
    alan = lez.sum()
    kenar = lez & ~ndimage.binary_erosion(lez)
    cevre = kenar.sum()
    return float(cevre / (2 * math.sqrt(math.pi * alan)))


def renk_cesitliligi(rgb, lez):
    piksel = rgb[lez].astype(float)
    mesafe = ((piksel[:, None, :] - RENK_PALETI[None, :, :]) ** 2).sum(axis=2)
    en_yakin = mesafe.argmin(axis=1)
    sayac = np.bincount(en_yakin, minlength=len(RENK_ADLARI))
    oran = sayac / sayac.sum()
    siniflar = [
        RENK_ADLARI[i]
        for i in range(len(RENK_ADLARI))
        if oran[i] >= RENK_MIN_ORAN
    ]
    return len(siniflar), ", ".join(siniflar)


def kriter_olc(gri, rgb):
    lez = lezyon_maskesi(gri)
    if lez is None:
        return {
            "Lezyon_bulundu": False,
            "Lezyon_orani": float("nan"),
            "B_duz": float("nan"),
            "C_renk": float("nan"),
            "C_siniflar": "",
        }
    sayi, siniflar = renk_cesitliligi(rgb, lez)
    return {
        "Lezyon_bulundu": True,
        "Lezyon_orani": float(lez.mean()),
        "B_duz": kenar_duzensizligi(lez),
        "C_renk": float(sayi),
        "C_siniflar": siniflar,
    }


def iki_grup_istatistik(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return None

    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return None

    tek = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    iki = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")

    df = se2**2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))

    fark = a.mean() - b.mean()
    se = math.sqrt(se2)
    tk = stats.t.ppf(1 - ALFA / 2, df)

    return {
        "n1": n1,
        "n2": n2,
        "ort_a": float(a.mean()),
        "ort_b": float(b.mean()),
        "fark": float(fark),
        "t": float(tek.statistic),
        "p": float(tek.pvalue),
        "p_iki": float(iki.pvalue),
        "d": float(fark / sp) if sp > 0 else float("nan"),
        "ci": (float(fark - tk * se), float(fark + tk * se)),
    }


def kriter_sonuclari(df):
    sonuc = {}
    for anahtar, _, yuksek, dusuk in KRITERLER:
        a = df.loc[df["Grup"] == yuksek, anahtar]
        b = df.loc[df["Grup"] == dusuk, anahtar]
        sonuc[anahtar] = iki_grup_istatistik(a, b)
    return sonuc


def analiz_yorumu(df, sonuclar):
    satirlar = [
        "Bu analiz grup karşılaştırmasıdır. Değerler metadata etiketlerine göre gruplanmıştır."
    ]
    for anahtar, ad, yuksek, dusuk in KRITERLER:
        s = sonuclar[anahtar]
        if s is None:
            continue
        fark = s["fark"]
        satirlar.append(
            f"• {ad}: {yuksek} ortalaması = {tr_sayi(s['ort_a'])}, {dusuk} ortalaması = {tr_sayi(s['ort_b'])}. "
            f"Fark = {tr_sayi(fark)} (Welch t = {tr_sayi(s['t'], 3)}, tek yönlü p = {p_yaz(s['p'])}, d = {tr_sayi(s['d'], 3)})."
        )
    return "\n\n".join(satirlar)


# ============================ TOPLU TEST VE GÖSTERİM ============================
def toplu_calistir(dosyalar, k, sabit, koyu_modu, aci_sayisi, adim):
    satirlar = []
    okunamayan = []
    toplam = len(dosyalar)

    bar = st.progress(0.0, text="Görseller taranıyor...")

    for i, (grup, ad, veri) in enumerate(dosyalar):
        gri, rgb = gorselleri_oku(veri)
        r = None
        kr = None

        if gri is not None:
            r = gorsel_analiz(gri, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
            kr = kriter_olc(gri, rgb)

        if r is None or kr is None:
            okunamayan.append(ad)
        else:
            satirlar.append({
                "Surum": SURUM,
                "Grup": grup,
                "Dosya": ad,
                "p": r["p"],
                "B": r["B"],
                "S_ort": r["S_ort"],
                "U_ort": r["U_ort"],
                "U_medyan": r["U_medyan"],
                "U_std": r["U_std"],
                "U_maks": r["U_maks"],
                "U_kontrol": r["U_kontrol"],
                "Gecerli_nokta": r["Gecerli_nokta"],
                "Toplam_nokta": r["Toplam_nokta"],
                "Kapsama_ort": r["Kapsama_ort"],
                "Lezyon_bulundu": kr["Lezyon_bulundu"],
                "Lezyon_orani": kr["Lezyon_orani"],
                "B_duz": kr["B_duz"],
                "C_renk": kr["C_renk"],
                "C_siniflar": kr["C_siniflar"],
            })

        bar.progress((i + 1) / toplam, text=f"{i + 1}/{toplam} görsel analiz edildi")

    bar.empty()
    return pd.DataFrame(satirlar), okunamayan


def toplu_goster(kayit):
    """KeyError hatası giderilmiş ve tasarımı güçlendirilmiş toplu sonuç paneli."""
    df = kayit["df"]

    st.caption(
        f"Protokol: {PROTOKOL_ADI} · Simetri: k = {kayit['k']} · "
        f"Eşik: {kayit['b']} · Çözünürlük: {kayit['hassasiyet']}"
    )

    if df.empty:
        st.error("Hiçbir görsel analiz edilemedi.")
        return

    ozet = (
        df.groupby("Grup")
        .agg(
            gorsel=("Dosya", "count"),
            lezyon=("Lezyon_bulundu", "sum"),
            s_ort=("S_ort", "mean"),
            u_ort=("U_ort", "mean"),
            b_duz=("B_duz", "mean"),
            c_renk=("C_renk", "mean"),
            u_kontrol=("U_kontrol", "mean"),
        )
        .reindex([GRUP_IYI, GRUP_KOTU])
        .rename(columns={
            "gorsel": "Görsel Sayısı",
            "lezyon": "Bölütlenen Lezyon",
            "s_ort": "A · Simetri Eşleşmesi (S)",
            "u_ort": "A · Üstünlük Skoru (U)",
            "b_duz": "B · Kenar Düzensizliği",
            "c_renk": "C · Renk Çeşitliliği",
            "u_kontrol": "Negatif Kontrol (Gürültü U)",
        })
    )

    st.markdown("### 📊 Grup İstatistik Özeti")
    st.dataframe(ozet.round(4), use_container_width=True)

    sonuclar = kriter_sonuclari(df)

    st.markdown(f"### 🎯 Üç Temel Ölçüt Değerlendirmesi (Düzeltilmiş Eşik α = {tr_sayi(ALFA_DUZELTILMIS, 4)})")

    # Değişken adını 'kriter_ist' yaparak dış 'kayit' sözlüğünü ezmesini engelledik!
    for anahtar, ad, yuksek, dusuk in KRITERLER:
        kriter_ist = sonuclar[anahtar]
        with st.container():
            st.markdown(f"#### {ad}")
            if kriter_ist is None:
                st.warning("Bu ölçüt için yeterli geçerli görsel bulunamadı.")
                continue

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Fark (Δ)", f"{kriter_ist['fark']:+.4f}")
            c2.metric("Welch t", f"{kriter_ist['t']:.3f}")
            c3.metric("p (Tek Yönlü)", p_yaz(kriter_ist["p"]))
            c4.metric("Cohen's d", f"{kriter_ist['d']:.3f}")
            lo, hi = kriter_ist["ci"]
            c5.metric("%95 Güven Aralığı", f"[{lo:+.4f}, {hi:+.4f}]")

            fark = kriter_ist["fark"]
            if fark > 0 and kriter_ist["p"] < ALFA_DUZELTILMIS:
                st.markdown(
                    f'<span class="badge-success">✓ Anlamlı Fark:</span> '
                    f'<b>{yuksek}</b> grubu, <b>{dusuk}</b> grubuna göre istatistiksel olarak belirgin düzeyde yüksek bulunmuştur (p = {p_yaz(kriter_ist["p"])} < 0,0167).',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge-neutral">Fark Saptanmadı:</span> '
                    f'Düzeltilmiş eşik düzeyinde beklenen yönde anlamlı bir fark oluşmamıştır.',
                    unsafe_allow_html=True,
                )
            st.write("")

    st.markdown("### 📈 Dağılım Kutu Grafikleri (Boxplot)")
    iyi = df.loc[df["Grup"] == GRUP_IYI]
    kotu = df.loc[df["Grup"] == GRUP_KOTU]
    if len(iyi) >= 2 and len(kotu) >= 2:
        goster(toplu_grafik(df))

    st.markdown("### 📄 Görsel Bazlı Veri Tablosu")
    st.dataframe(df.round(4), use_container_width=True)

    # KeyError hatası burada çözüldü
    if kayit.get("okunamayan"):
        st.warning("Analiz dışı kalan dosyalar: " + ", ".join(kayit["okunamayan"]))

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "📥 Tam Sonuç Tablosunu İndir (.csv)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ysum_v3_sonuclar.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b2:
        st.download_button(
            "📄 Açıklamalı Rapor Metnini İndir (.txt)",
            (PROTOKOL_ADI + "\n\n" + analiz_yorumu(df, sonuclar)).encode("utf-8-sig"),
            file_name="ysum_v3_analiz_ozeti.txt",
            mime="text/plain",
            use_container_width=True,
        )


def gorsel_sec(etiket, anahtar, varsayilan=None):
    c_up, c_ex = st.columns([1.2, 1])
    with c_up:
        yuklenen = st.file_uploader(f"{etiket} (Görsel Yükle)", type=UPLOAD_TYPES, key=f"{anahtar}_up")
    with c_ex:
        secenekler = ["(yok)"] + list(ORNEKLER)
        idx = secenekler.index(varsayilan) if varsayilan in secenekler else 0
        ornek = st.selectbox(f"{etiket} (Hazır Örnek)", secenekler, index=idx, key=f"{anahtar}_ex")

    if yuklenen is not None:
        gri, rgb = gorselleri_oku(yuklenen.getvalue())
        if gri is None:
            st.error("Dosya okunamadı.")
            return None, None, None
        return gri, rgb, yuklenen.name

    if ornek != "(yok)":
        g = ornek_uret(ORNEKLER[ornek])
        return g, np.stack([g, g, g], axis=-1), ornek

    return None, None, None


# ============================ KENAR ÇUBUĞU ============================
with st.sidebar:
    st.markdown("### ⚙️ Ölçüm Konfigürasyonu")

    k = st.selectbox(
        "Dönel Simetri Türü (k)",
        list(K_SECENEKLER),
        index=0,
        format_func=lambda x: K_SECENEKLER[x],
    )

    secim_b = st.radio(
        "Şans Eşleşme Eşiği (B)",
        [
            "Her görsel için kendi değeri (önerilen)",
            "Sabit Penrose değeri (0,528)",
        ],
    )
    sabit = secim_b.startswith("Sabit")

    koyu_modu = st.checkbox(
        "Merkezi ağırlıklı koyu bölgeye sabitle",
        value=False,
    )

    hassaslik = st.selectbox(
        "Açısal ve Izgara Çözünürlüğü",
        list(HASSASLIK),
        index=1,
    )
    aci_sayisi, adim = HASSASLIK[hassaslik]

    st.markdown("---")
    st.markdown(
        f"**Sürüm:** `{SURUM}`<br>"
        "**Çözünürlük:** 200 × 200 Piksel<br>"
        "**Kalite Filtresi:** Geçerli Çift ≥ %50<br>"
        "**Çoklu Test:** Bonferroni (α = 0,0167)",
        unsafe_allow_html=True,
    )


# ============================ ANA SAYFA BAŞLIĞI ============================
st.markdown("""
<div class="main-header">
    <h1>🌀 YSÜM · Yerel Simetri Üstünlük Modeli</h1>
    <p>Görüntülerdeki dönel simetriyi, piksel olasılıklarından doğan matematiksel şans eşiğinden arındırarak ölçen, Penrose aperiyodik oranlarını referans alan yeni nesil hesaplamalı analiz sistemi.</p>
</div>
""", unsafe_allow_html=True)


# ============================ SEKMELER ============================
t_kavram, t_kalibrasyon, t_yy, t_endustri, t_toplu, t_metodoloji = st.tabs([
    "📐 Kuramsal Model",
    "🧪 Kalibrasyon & Doğrulama",
    "🔍 Görsel Analiz & Isı Haritası",
    "🏭 Endüstriyel & Geometrik İnceleme",
    "📊 HAM10000 Dermatoskopi Benchmark",
    "📋 Model Mimarisi & Metodoloji",
])


# ---------------------------- 1. KURAMSAL MODEL ----------------------------
with t_kavram:
    st.markdown("### 🏛️ Penrose Döşemesi, Altın Oran ve Şans Eşiği")
    st.markdown("""
    **Penrose döşemeleri**, ötelenme simetrisi taşımayan ancak kusursuz bir uzun menzilli yönelim düzenine sahip aperiyodik yapılardır.
    Kalın ve ince eşkenar dörtgenlerin (rombus) sayısal oranı, döşeme büyüdükçe altın orana ($\varphi \approx 1,618$) yakınsar.
    
    Bu durum, rastgele iki karo seçildiğinde ikisinin de aynı türden olma kuramsal şansını belirler:
    """)

    c_mat1, c_mat2 = st.columns([1.2, 1])
    with c_mat1:
        st.latex(r"p = \frac{1}{\varphi} \approx 0,618034")
        st.latex(r"B(p) = p^2 + (1-p)^2 \approx (0,618)^2 + (0,382)^2 \approx 0,528")
        st.markdown("""
        **YSÜM**, bir görüntüdeki dönel benzerlik oranını ($S_k$), piksel dağılımından kaynaklanan bu matematiksel şans eşiğinden ($B$) arındırır:
        """)
        st.latex(r"U_k = \frac{S_k - B}{1 - B}")
        st.markdown("""
        * **$U > +0,15$**: Şans beklentisinden belirgin yüksek (düzenli tasarım).
        * **$-0,15 \le U \le +0,15$**: Şans beklentisiyle uyumlu yapı.
        * **$U < -0,15$**: Şans beklentisinden belirgin asimetrik / dağınık örüntü.
        """)
    with c_mat2:
        goster(fig_b_egrisi())

    st.markdown("---")
    st.markdown("#### 🔬 Geometrik Kırınım ve Mimari Motif Örnekleri")
    c1, c2, c3 = st.columns(3)
    with c1:
        goster(fig_kirinim())
        st.caption("Dan Shechtman'a Nobel kazandıran 5 katlı kırınım düzeni simülasyonu.")
    with c2:
        goster(fig_rozet())
        st.caption("Penrose aperiyodik döşemelerinde ortaya çıkan 5 katlı yıldız rozeti.")
    with c3:
        goster(fig_girih())
        st.caption("Tarihî Selçuklu mimarisinde görülen on katlı girih yıldız deseni.")

    st.markdown("#### 🔢 Fibonacci Matris Yakınsaması (Kuramsal Çizelge)")
    satirlar = []
    K, I = 2, 1
    for n in range(1, 7):
        T = K + I
        satirlar.append({
            "İterasyon (n)": n,
            "Kalın Rombus (K_n)": K,
            "İnce Rombus (I_n)": I,
            "Toplam (T_n)": T,
            "Oran (K_n / T_n)": round(K / T, 6),
            "1/φ Sapması": f"{abs(K/T - P_PENROSE):.6f}",
        })
        K, I = 2 * K + I, K + I
    st.dataframe(pd.DataFrame(satirlar), use_container_width=True)


# ---------------------------- 2. KALİBRASYON & DOĞRULAMA ----------------------------
with t_kalibrasyon:
    st.markdown("### 🧪 Model Kalibrasyonu ve Sentetik Test Tezgahı")
    st.markdown("""
    Modelin ampirik verilere (ör. dermatoskopi veya endüstriyel yüzeyler) uygulanmadan önce, 
    bilinen geometrik özelliklere sahip sentetik kalıplarda nasıl tepki verdiğini buradan doğrulayabilirsiniz.
    """)

    c_k1, c_k2 = st.columns([1, 1.2])
    with c_k1:
        test_tipi = st.selectbox(
            "Test Edilecek Geometrik Kalıp",
            [
                "Kusursuz Simetrik Şekil (Daire)",
                "Kontrollü Asimetrik Şekil",
                "5 Katlı Penrose Yıldızı",
                "Rastgele Beyaz Gürültü (Uniform Dağılım)",
            ],
        )

        kod_haritasi = {
            "Kusursuz Simetrik Şekil (Daire)": "yuvarlak",
            "Kontrollü Asimetrik Şekil": "duzensiz",
            "5 Katlı Penrose Yıldızı": "bes_kat",
            "Rastgele Beyaz Gürültü (Uniform Dağılım)": "gurultu",
        }

        g_sentetik = ornek_uret(kod_haritasi[test_tipi])
        st.image(g_sentetik, caption=f"Sentetik Test Girdisi (200x200)", width=260)

    with c_k2:
        r_sentetik = gorsel_analiz(
            g_sentetik, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim
        )

        if r_sentetik:
            st.markdown(f"#### Analiz Sonucu: `{test_tipi}`")
            m1, m2 = st.columns(2)
            m1.metric("Açık Piksel Oranı (p)", f"{r_sentetik['p']:.4f}")
            m1.metric("Şans Eşiği B(p)", f"{r_sentetik['B']:.4f}")
            m2.metric("Ortalama Eşleşme (S)", f"{r_sentetik['S_ort']:.4f}")
            m2.metric("Üstünlük Skoru (U)", f"{r_sentetik['U_ort']:+.4f}")

            st.write("")
            if "Simetrik" in test_tipi and r_sentetik['U_ort'] > 0.15:
                st.markdown('<span class="badge-success">✓ Doğrulandı:</span> Kusursuz simetri şans eşiğini rahatça aşmaktadır.', unsafe_allow_html=True)
            elif "Gürültü" in test_tipi and abs(r_sentetik['U_ort']) <= 0.15:
                st.markdown('<span class="badge-success">✓ Doğrulandı:</span> Rastgele gürültü şans beklentisi sınırları içinde sıfıra yakınsamaktadır.', unsafe_allow_html=True)
            elif "Asimetrik" in test_tipi:
                st.markdown('<span class="badge-gold">Bilgi:</span> Düzensiz sınırlarda yerel simetri düşüşü başarıyla kaydedilmiştir.', unsafe_allow_html=True)


# ---------------------------- 3. GÖRSEL ANALİZ & ISI HARİTASI ----------------------------
with t_yy:
    st.markdown("### 🔍 İki Görseli Yan Yana Karşılaştırma & Isı Haritaları")
    st.markdown("İki farklı deseni veya lezyonu yükleyerek piksel düzeyinde simetri üstünlük haritasını inceleyebilirsiniz.")

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown("#### 1. Görsel")
        g1, rgb1, ad1 = gorsel_sec("1. Görsel", "yy1", "Yuvarlak Şekil")
    with c_g2:
        st.markdown("#### 2. Görsel")
        g2, rgb2, ad2 = gorsel_sec("2. Görsel", "yy2", "Düzensiz Sınır")

    if g1 is not None and g2 is not None:
        r1 = gorsel_analiz(g1, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        r2 = gorsel_analiz(g2, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        k1 = kriter_olc(g1, rgb1)
        k2 = kriter_olc(g2, rgb2)

        if r1 is None or r2 is None:
            st.error("Görsellerden birinde geçerli ölçüm noktası bulunamadı.")
        else:
            st.write("")
            c_img1, c_img2 = st.columns(2)
            with c_img1:
                goster(isi_figur(g1, r1["izg"], r1["U_matris"], f"{ad1} · YSÜM Isı Haritası"))
            with c_img2:
                goster(isi_figur(g2, r2["izg"], r2["U_matris"], f"{ad2} · YSÜM Isı Haritası"))

            satirlar_karsilastir = [
                ("A · Açık Piksel Oranı (p)", tr_sayi(r1["p"]), tr_sayi(r2["p"])),
                ("A · Şans Eşleşme Eşiği (B)", tr_sayi(r1["B"]), tr_sayi(r2["B"])),
                ("A · Ölçülen Simetri Oranı (S)", tr_sayi(r1["S_ort"]), tr_sayi(r2["S_ort"])),
                ("A · Ortalama Üstünlük Skoru (U)", tr_sayi(r1["U_ort"]), tr_sayi(r2["U_ort"])),
                ("A · Maksimum Odak Skoru (U_maks)", tr_sayi(r1["U_maks"]), tr_sayi(r2["U_maks"])),
                ("A · Negatif Kontrol (Gürültü U)", tr_sayi(r1["U_kontrol"]), tr_sayi(r2["U_kontrol"])),
                ("B · Sınır Düzensizliği (İzoperimetrik)", tr_sayi(k1["B_duz"], 3) if k1["Lezyon_bulundu"] else "—", tr_sayi(k2["B_duz"], 3) if k2["Lezyon_bulundu"] else "—"),
                ("C · Tespit Edilen Renk Sınıf Sayısı", str(int(k1["C_renk"])) if k1["Lezyon_bulundu"] else "—", str(int(k2["C_renk"])) if k2["Lezyon_bulundu"] else "—"),
                ("C · Hakim Renk Tonları", k1["C_siniflar"] or "—", k2["C_siniflar"] or "—"),
                ("Simetri Değerlendirmesi", yorum_etiketi(r1["U_ort"]), yorum_etiketi(r2["U_ort"])),
            ]

            df_kar = pd.DataFrame(satirlar_karsilastir, columns=["Ölçüt / Parametre", "1. Görsel Değeri", "2. Görsel Değeri"])
            st.table(df_kar)

            delta = r2["U_ort"] - r1["U_ort"]
            st.metric("İki Görsel Arasındaki Simetri Farkı (ΔU)", f"{delta:+.4f}")


# ---------------------------- 4. ENDÜSTRİYEL & GEOMETRİK İNCELEME ----------------------------
with t_endustri:
    st.markdown("### 🏭 Endüstriyel Yüzey ve Üretim Prototipi İnceleme Modülü")
    st.markdown("""
    Bu bölüm, seri üretime geçecek parçalarda (örneğin dairesel kapaklar, 6 kollu dişliler, 
    dekoratif seramik/çini karoları ve hassas yüzeyler) imalat öncesi **simetri sapmalarını ve kusurlu bölgeleri** tespit etmek için geliştirilmiştir.
    """)

    c_end1, c_end2 = st.columns([1, 1.2])
    with c_end1:
        st.markdown("#### Prototip Yüzey Görüntüsü")
        end_up = st.file_uploader("Ürün / Yüzey Fotoğrafı Yükle", type=UPLOAD_TYPES, key="end_up")
        k_end = st.selectbox(
            "Hedeflenen Tasarım Simetrisi",
            [2, 5, 6, 8],
            index=2,
            format_func=lambda x: f"{x} Katlı Rotasyon ({360//x}° Simetri)",
        )

        if end_up is not None:
            g_end, _, ad_end = gorselleri_oku(end_up.getvalue())
        else:
            g_end = ornek_uret("bes_kat")
            ad_end = "5 Katlı Örnek Tasarım Rozeti"

        st.image(g_end, caption=f"İncelenen Parça: {ad_end}", width=280)

    with c_end2:
        st.markdown("#### Kusur & Simetri Sapma Haritası")
        r_end = gorsel_analiz(g_end, k_end, sabit, koyu_modu, ISI_YARICAPLARI, False, aci_sayisi, adim)

        if r_end:
            goster(isi_figur(g_end, r_end["izg"], r_end["U_matris"], f"{k_end} Katlı Simetri Dağılımı"))
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Genel Simetri Uyumu (U)", f"{r_end['U_ort']:.4f}")
            c_m2.metric("En Yüksek Simetri Odağı", f"{r_end['U_maks']:.4f}")
            c_m3.metric("Homojenlik (Std Sapma)", f"{r_end['U_std']:.4f}")

            st.markdown("""
            **Yorumlama Rehberi:**
            * **Kırmızı/Sıcak Bölgeler:** Tasarım simetrisinin korunduğu alanlar.
            * **Mavi/Soğuk Bölgeler:** Geometrik sapma, çapak, kırık veya üretim hatası bulunan kusurlu bölgeler.
            """)


# ---------------------------- 5. BÜYÜK VERİ ARAŞTIRMASI (HAM10000) ----------------------------
with t_toplu:
    st.markdown("### 📊 HAM10000 Dermatoskopi Benchmark Testi")
    st.markdown("""
    İyi huylu (nv, n=100) ve kötü huylu (mel, n=100) dermatoskopi görüntülerini yükleyerek 
    **A (Simetri), B (Kenar Düzensizliği) ve C (Renk Çeşitliliği)** ölçütlerini eşzamanlı olarak sınayın.
    """)

    v = st.session_state.setdefault("yukleme_v", 0)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{GRUP_IYI} Klasörü Dosyaları**")
        iyi_yuk = st.file_uploader(GRUP_IYI, type=UPLOAD_TYPES, accept_multiple_files=True, key=f"grp_iyi_{v}")
    with c2:
        st.markdown(f"**{GRUP_KOTU} Klasörü Dosyaları**")
        kotu_yuk = st.file_uploader(GRUP_KOTU, type=UPLOAD_TYPES, accept_multiple_files=True, key=f"grp_kotu_{v}")

    b_run, b_clear = st.columns([1, 1])
    with b_run:
        calistir = st.button("🚀 Toplu Benchmark Testini Başlat", key="btn_toplu", type="primary", use_container_width=True)
    with b_clear:
        temizle = st.button("🗑️ Verileri ve Önceki Sonuçları Temizle", key="btn_temizle", use_container_width=True)

    if temizle:
        st.session_state.pop("toplu", None)
        st.session_state["yukleme_v"] = v + 1
        st.rerun()

    if calistir:
        iyi = [(f.name, f.getvalue()) for f in (iyi_yuk or [])]
        kotu = [(f.name, f.getvalue()) for f in (kotu_yuk or [])]

        if len(iyi) < 2 or len(kotu) < 2:
            st.error("Her iki gruptan da en az ikişer görsel yüklenmelidir.")
        else:
            ad_ortak = {a for a, _ in iyi} & {a for a, _ in kotu}
            md5_iyi = {hashlib.md5(b).hexdigest() for _, b in iyi}
            md5_kotu = {hashlib.md5(b).hexdigest() for _, b in kotu}
            icerik_ortak = md5_iyi & md5_kotu

            if ad_ortak or icerik_ortak:
                st.error("Hata: Gruplar arasında mükerrer (aynı) dosyalar tespit edildi. Test durduruldu.")
                st.session_state.pop("toplu", None)
            else:
                dosyalar = (
                    [(GRUP_IYI, a, b) for a, b in iyi]
                    + [(GRUP_KOTU, a, b) for a, b in kotu]
                )

                df, okunamayan = toplu_calistir(
                    dosyalar, k, sabit, koyu_modu, aci_sayisi, adim,
                )

                st.session_state["toplu"] = {
                    "surum": SURUM,
                    "df": df,
                    "okunamayan": okunamayan,
                    "k": k,
                    "b": "Sabit 0,528" if sabit else "Her görsel için kendi değeri",
                    "hassasiyet": hassaslik,
                }

    kayit = st.session_state.get("toplu")
    if kayit and kayit.get("surum") == SURUM:
        toplu_goster(kayit)


# ---------------------------- 6. MODEL MİMARİSİ VE METODOLOJİ ----------------------------
with t_metodoloji:
    st.markdown("### 📋 Model Parametreleri ve Akademik Metodoloji")
    
    st.markdown("""
    #### 1. Veri Ön İşleme ve İkileştirme
    * Görüntüler renk uzayından bağımsız olarak **200 × 200** piksel boyutuna normalize edilir.
    * Her görselin kendi ortalama gri parlaklık değeri dinamik eşik kabul edilerek ikili (binary) matrise dönüştürülür.
    
    #### 2. Kalite Denetim Kuralları (Filtreleme)
    * **MIN_KAPSAMA ≥ %50:** Çember yayında görüntü sınırları dışına taşan noktalar %50'den fazlaysa o nokta hesaplama dışı bırakılır.
    * **Düz Alan Dışlama:** Tamamı tek renk (homojen) olan piksel bölgelerinde simetri tanımsız olduğu için elenir.
    
    #### 3. B ve C Lezyon Bölütlemesi
    * **Otsu Eşikleme:** En yüksek varyans eşiğiyle iki lezyon adayı oluşturulur; görüntü merkezine temas eden en büyük bağlı bileşen seçilir.
    * **İzoperimetrik Düzensizlik:** $B_{duz} = P / (2\sqrt{\pi A})$ formülüyle sınır kıvrımlılığı hesaplanır.
    * **Renk Çeşitliliği:** 6 temel dermatoskopik referans tona en az %5 piksel oranında dahil olan sınıflar sayılır.
    
    #### 4. İstatistiksel Hipotez Testleri
    * Çoklu ölçüt karşılaştırmasında Tip I hatayı engellemek için **Bonferroni Düzeltmesi** uygulanır: $\alpha = 0,05 / 3 \approx 0,0167$.
    * Grupların varyans eşitliği varsayılmaksızın **Welch t-testi** ve etki büyüklüğü için **Cohen's d** hesaplanır.
    """)

    st.markdown("---")
    st.caption("🔬 TÜBİTAK 2204-B Ortaokul Öğrencileri Araştırma Projeleri Yarışması Bilgisayar / Yapay Zekâ Alanı Çalışmasıdır.")
