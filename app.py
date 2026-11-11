import hashlib
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage, stats

st.set_page_config(
    page_title="YSÜM · Yerel Simetri Üstünlük Modeli",
    page_icon="🌀",
    layout="wide",
)

# ============================ PROTOKOL ============================
PHI = (1 + math.sqrt(5)) / 2
P_PENROSE = 1 / PHI
B_PENROSE = P_PENROSE**2 + (1 - P_PENROSE) ** 2

IMG_N = 200
ALFA = 0.05
KRITER_SAYISI = 3
ALFA_DUZELTILMIS = ALFA / KRITER_SAYISI  # Bonferroni, ≈ 0,0167
YORUM_ESIGI = 0.15
IZG_BASLANGIC = 10

# A (simetri) kalite kuralları: v2 ile aynı
MIN_KAPSAMA = 0.50
DUZ_ALANLARI_DISLA = True

# B ve C için lezyon bölütleme kuralları (sabit)
LEZYON_MIN_ORAN = 0.02
LEZYON_MAX_ORAN = 0.80
RENK_MIN_ORAN = 0.05

SURUM = "YSÜM-v3-ABC"
PROTOKOL_ADI = (
    "YSÜM-v3 · A: simetri (k = 2, kapsama ≥ 0,50, düz alan dışlama) · "
    "B: kenar düzensizliği · C: renk çeşitliliği · "
    "düzeltilmiş α = 0,05 / 3 ≈ 0,0167"
)

GRUP_IYI = "İyi huylu (nv)"
GRUP_KOTU = "Kötü huylu (mel)"

# (anahtar, ad, beklenen yüksek grup, beklenen düşük grup)
# Yönler önceden belirlenmiştir ve sonuca göre değiştirilmez.
KRITERLER = [
    ("U_ort", "A · Simetri (U_ort)", GRUP_IYI, GRUP_KOTU),
    ("B_duz", "B · Kenar düzensizliği", GRUP_KOTU, GRUP_IYI),
    ("C_renk", "C · Renk çeşitliliği (renk sınıfı sayısı)", GRUP_KOTU, GRUP_IYI),
]

UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp"]

K_SECENEKLER = {
    2: "2 kat · 180° (dermatoskopi)",
    5: "5 kat · 72° (Penrose)",
    6: "6 kat · 60°",
    8: "8 kat · 45°",
}

SKOR_YARICAPLARI = (20, 40, 60, 80)
ISI_YARICAPLARI = (10, 20, 30, 40)

HASSASLIK = {
    "Standart (36 açı, adım 8)": (36, 8),
    "Hassas (180 açı, adım 4)": (180, 4),
    "Çok hassas (360 açı, adım 4)": (360, 4),
}

ORNEKLER = {
    "Yuvarlak şekil": "yuvarlak",
    "Düzensiz şekil": "duzensiz",
    "5 katlı desen": "bes_kat",
    "Rastgele gürültü": "gurultu",
}

# C için yaklaşık dermatoskopik renk paleti (RGB). Sabit ve önceden tanımlı.
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


# ============================ BİÇİMLENDİRME ============================
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
        return "Tesadüften belirgin yüksek"
    if u < -YORUM_ESIGI:
        return "Tesadüften belirgin düşük"
    return "Tesadüf beklentisine yakın"


def etki_etiketi(d):
    a = abs(d)
    if a < 0.20:
        return "çok küçük"
    if a < 0.50:
        return "küçük"
    if a < 0.80:
        return "orta"
    return "büyük"


# ============================ ÇİZİMLER ============================
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
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(z, cmap="magma")
    ax.set_title("Beşli simetrili örüntü (illüstrasyon)", fontsize=9)
    ax.axis("off")
    return fig


def fig_rozet():
    aci = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
    x, y = np.cos(aci), np.sin(aci)
    fig, ax = plt.subplots(figsize=(4, 4))
    for i in range(5):
        j = (i + 2) % 5
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#1f4e79", lw=2)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="#8aa6c1", lw=0.8)
    ax.scatter(x, y, s=30, color="#c0392b", zorder=3)
    ax.set_aspect("equal")
    ax.set_title("5 katlı rozet", fontsize=9)
    ax.axis("off")
    return fig


def fig_girih():
    aci = 2 * np.pi * np.arange(10) / 10
    x, y = np.cos(aci), np.sin(aci)
    fig, ax = plt.subplots(figsize=(4, 4))
    for i in range(10):
        j = (i + 3) % 10
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#6b4f1d", lw=1.4)
    ax.set_aspect("equal")
    ax.set_title("On katlı yıldız çizgisi (girih benzeri)", fontsize=9)
    ax.axis("off")
    return fig


def fig_b_egrisi():
    p = np.linspace(0, 1, 300)
    b = p**2 + (1 - p) ** 2
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(p, b, color="#1f4e79")
    ax.axvline(P_PENROSE, ls="--", color="#c0392b")
    ax.axhline(B_PENROSE, ls="--", color="#c0392b")
    ax.scatter([P_PENROSE], [B_PENROSE], color="#c0392b", zorder=3)
    ax.set_xlabel("Açık piksel oranı p")
    ax.set_ylabel("Şans eşiği B(p)")
    ax.set_title("B(p) = p² + (1−p)²; p = 1/φ için B ≈ 0,528", fontsize=10)
    ax.grid(alpha=0.3)
    return fig


def isi_figur(gri, izg, u_matris, baslik):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(gri, cmap="gray", extent=(0, IMG_N, IMG_N, 0))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad((0, 0, 0, 0))
    maskeli = np.ma.masked_invalid(u_matris)

    e = (izg[0], izg[-1], izg[-1], izg[0])
    im = ax.imshow(
        maskeli, cmap=cmap, vmin=-0.5, vmax=0.5, alpha=0.65, extent=e
    )
    ax.set_xlim(0, IMG_N)
    ax.set_ylim(IMG_N, 0)
    ax.set_title(baslik, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def toplu_grafik(df):
    fig, eksen = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (anahtar, ad, _, _) in zip(eksen, KRITERLER):
        veri = [
            df.loc[df["Grup"] == g, anahtar].dropna().values
            for g in (GRUP_IYI, GRUP_KOTU)
        ]
        ax.boxplot(veri, widths=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["İyi huylu\n(nv)", "Kötü huylu\n(mel)"])
        if anahtar == "U_ort":
            ax.axhline(0, color="gray", lw=0.8)
            ax.axhline(YORUM_ESIGI, color="#c0392b", ls=":", lw=0.8)
            ax.axhline(-YORUM_ESIGI, color="#c0392b", ls=":", lw=0.8)
        ax.set_title(ad, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ============================ GÖRÜNTÜ OKUMA ============================
def gorselleri_oku(veri):
    """Bayt verisinden gri (ölçüm için) ve RGB (renk için) dizileri döndürür."""
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
        g = np.where(r < 60, 210, 40)
    elif kod == "duzensiz":
        sinir = 55 + 12 * np.sin(3 * th + 0.5) + 6 * np.cos(5 * th)
        g = np.where(r < sinir, 200, 45)
    elif kod == "bes_kat":
        g = np.where((r < 80) & (np.cos(5 * th) > 0), 220, 40)
    else:
        g = np.random.default_rng(1).integers(0, 256, size=(IMG_N, IMG_N))

    return g.astype(np.uint8)


def merkez_bul(binary):
    koyu = ~binary
    if 0.02 <= koyu.mean() <= 0.98:
        ys, xs = np.nonzero(koyu)
        return int(round(xs.mean())), int(round(ys.mean()))
    return IMG_N // 2, IMG_N // 2


# ============================ A: SİMETRİ (v2 ile aynı) ============================
def sim_hesapla(binary, k, cxs, cys, radii, aci_sayisi, parca=128):
    """S değerleri ve geometrik kapsama. Kurallara uymayan noktalar NaN döner."""
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


# ============================ B ve C: LEZYON ÖLÇÜTLERİ ============================
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
    """
    Otsu eşiğiyle iki ikili maske üretir. Her ikisinde de bağlı bileşenlerden
    alan oranı kurala uyanlar aday olur. Merkeze değen ve en büyük aday seçilir.
    Aday yoksa None döner (lezyon bulunamadı).
    """
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
    """P / (2√(πA)). Daire yaklaşık 1; değer büyüdükçe sınır daha düzensiz."""
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


# ============================ İSTATİSTİK ============================
def iki_grup_istatistik(a, b):
    """
    a: beklenen olarak daha yüksek grup, b: diğer grup.
    Tek yönlü Welch testi H1: μ_a > μ_b. fark = ort(a) − ort(b).
    """
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


def kriter_metni(anahtar, ad, yuksek, dusuk, s):
    baslik = f"**{ad}** · beklenen yön: {yuksek} > {dusuk}"
    if s is None:
        return baslik + "\n\nBu ölçüt için her grupta yeterli geçerli görsel yok."

    fark = s["fark"]
    lo, hi = s["ci"]

    if fark > 0:
        yon = f"{yuksek} grubunun ortalaması, {dusuk} grubundan {tr_sayi(fark)} birim yüksek."
    else:
        yon = f"{yuksek} grubunun ortalaması, {dusuk} grubundan {tr_sayi(-fark)} birim düşük."

    satirlar = [
        baslik,
        f"Ortalamalar: {yuksek} = {tr_sayi(s['ort_a'])}, "
        f"{dusuk} = {tr_sayi(s['ort_b'])}. {yon}",
        f"Welch t = {tr_sayi(s['t'], 3)}; tek yönlü p = {p_yaz(s['p'])}; "
        f"iki yönlü p = {p_yaz(s['p_iki'])}; düzeltilmiş eşik = "
        f"{tr_sayi(ALFA_DUZELTILMIS, 4)}.",
        f"Cohen's d = {tr_sayi(s['d'], 3)} ({etki_etiketi(s['d'])}); "
        f"%95 güven aralığı [{tr_sayi(lo)}, {tr_sayi(hi)}].",
    ]

    if fark > 0 and s["p"] < ALFA_DUZELTILMIS:
        satirlar.append(
            "**Karar:** Beklenen yönde ve düzeltilmiş eşikte anlamlıdır. "
            "Bu, bu örneklemde gözlenen bir istatistiksel farktır; tanı kanıtı değildir."
        )
    else:
        satirlar.append(
            "**Karar:** Düzeltilmiş eşikte beklenen yönde anlamlı bir fark yoktur."
        )

    if anahtar == "U_ort" and abs(s["ort_a"]) <= YORUM_ESIGI and abs(s["ort_b"]) <= YORUM_ESIGI:
        satirlar.append(
            "Her iki ortalama ±0,15 yorum konvansiyonunun içindedir; "
            "‘tesadüften belirgin yüksek simetri’ yorumu yapılmaz."
        )

    return "\n\n".join(satirlar)


def genel_sonuc(sonuclar):
    basarili = [
        ad
        for anahtar, ad, _, _ in KRITERLER
        if sonuclar[anahtar] is not None
        and sonuclar[anahtar]["fark"] > 0
        and sonuclar[anahtar]["p"] < ALFA_DUZELTILMIS
    ]
    if basarili:
        return (
            "Beklenen yönde ve düzeltilmiş eşikte anlamlı çıkan ölçüt(ler): "
            + ", ".join(basarili)
            + ". Diğer ölçütler için ayrı karar yukarıdadır."
        )
    return (
        "Üç ölçütün hiçbiri, önceden belirlenen yönde ve düzeltilmiş eşikte "
        "anlamlı değildir."
    )


def analiz_yorumu(df, sonuclar):
    satirlar = [
        "**Bu açıklama grup karşılaştırmasıdır.** İyi huylu (nv) ve kötü huylu (mel) "
        "adları yüklenen metadata gruplarından gelir. YSÜM tek bir görseli sınıflandırmaz."
    ]
    for anahtar, ad, yuksek, dusuk in KRITERLER:
        satirlar.append(kriter_metni(anahtar, ad, yuksek, dusuk, sonuclar[anahtar]))

    satirlar.append("**Genel sonuç:** " + genel_sonuc(sonuclar))
    satirlar.append(
        "**Sınır:** Bu analiz tanı, duyarlılık veya özgüllük vermez. "
        "Tıbbi karar için kullanılamaz."
    )
    return "\n\n".join(satirlar)


# ============================ TOPLU ANALİZ ============================
def toplu_calistir(dosyalar, k, sabit, koyu_modu, aci_sayisi, adim):
    satirlar = []
    okunamayan = []
    toplam = len(dosyalar)

    bar = st.progress(0.0, text="Görseller analiz ediliyor...")

    for i, (grup, ad, veri) in enumerate(dosyalar):
        gri, rgb = gorselleri_oku(veri)
        r = None
        kr = None

        if gri is not None:
            r = gorsel_analiz(
                gri, k, sabit, koyu_modu,
                SKOR_YARICAPLARI, True, aci_sayisi, adim,
            )
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

        bar.progress((i + 1) / toplam, text=f"{i + 1}/{toplam} görsel")

    bar.empty()
    return pd.DataFrame(satirlar), okunamayan


def toplu_goster(s):
    df = s["df"]

    st.caption(
        f"Protokol: {PROTOKOL_ADI} · k = {s['k']} · "
        f"B: {s['b']} · hassasiyet: {s['hassasiyet']}"
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
            "gorsel": "Görsel sayısı",
            "lezyon": "Lezyon bulunan",
            "s_ort": "A · Ortalama S",
            "u_ort": "A · Ortalama U_ort",
            "b_duz": "B · Ortalama düzensizlik",
            "c_renk": "C · Ortalama renk sınıfı",
            "u_kontrol": "Negatif kontrol (U_kontrol)",
        })
    )

    st.markdown("#### Grup özeti")
    st.dataframe(ozet.round(4))

    sonuclar = kriter_sonuclari(df)

    st.markdown(f"#### Üç ölçüt · düzeltilmiş eşik {tr_sayi(ALFA_DUZELTILMIS, 4)}")
    for anahtar, ad, yuksek, dusuk in KRITERLER:
        s = sonuclar[anahtar]
        st.markdown(f"**{ad}** · beklenen: {yuksek} > {dusuk}")
        if s is None:
            st.warning("Bu ölçüt için her grupta en az iki geçerli görsel gerekir.")
            continue
        m = st.columns(5)
        m[0].metric("Fark", f"{s['fark']:+.4f}")
        m[1].metric("Welch t", f"{s['t']:.3f}")
        m[2].metric("p (tek yönlü)", p_yaz(s["p"]))
        m[3].metric("Cohen's d", f"{s['d']:.3f}")
        lo, hi = s["ci"]
        m[4].metric("%95 güven aralığı", f"[{lo:+.4f}, {hi:+.4f}]")

    st.markdown("#### Açıklamalı analiz")
    st.markdown(analiz_yorumu(df, sonuclar))

    st.download_button(
        "📄 Açıklamalı analiz özetini indir",
        (PROTOKOL_ADI + "\n\n" + analiz_yorumu(df, sonuclar)).encode("utf-8-sig"),
        file_name="ysum_v3_analiz_ozeti.txt",
        mime="text/plain",
    )

    iyi = df.loc[df["Grup"] == GRUP_IYI]
    kotu = df.loc[df["Grup"] == GRUP_KOTU]
    if len(iyi) >= 2 and len(kotu) >= 2:
        goster(toplu_grafik(df))

    st.markdown("#### Görsel bazlı sonuçlar")
    st.dataframe(df.round(4))

    if s["okunamayan"]:
        st.warning("Analiz edilemeyen dosyalar: " + ", ".join(s["okunamayan"]))

    st.download_button(
        "📥 ysum_v3_sonuclar.csv indir",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ysum_v3_sonuclar.csv",
        mime="text/csv",
    )


# ============================ GİRDİ ============================
def gorsel_sec(etiket, anahtar, varsayilan=None):
    """Döndürür: (gri, rgb, ad). Yoksa (None, None, None)."""
    yuklenen = st.file_uploader(
        f"{etiket} (yükle)",
        type=UPLOAD_TYPES,
        key=f"{anahtar}_up",
    )

    secenekler = ["(yok)"] + list(ORNEKLER)
    idx = secenekler.index(varsayilan) if varsayilan in secenekler else 0

    ornek = st.selectbox(
        f"{etiket} (örnek)",
        secenekler,
        index=idx,
        key=f"{anahtar}_ex",
    )

    if yuklenen is not None:
        gri, rgb = gorselleri_oku(yuklenen.getvalue())
        if gri is None:
            st.error("Bu dosya okunamadı.")
            return None, None, None
        return gri, rgb, yuklenen.name

    if ornek != "(yok)":
        g = ornek_uret(ORNEKLER[ornek])
        return g, np.stack([g, g, g], axis=-1), ornek

    return None, None, None


# ============================ KENAR ÇUBUĞU ============================
with st.sidebar:
    st.header("⚙️ Ayarlar")

    k = st.selectbox(
        "Simetri türü (k)",
        list(K_SECENEKLER),
        index=0,
        format_func=lambda x: K_SECENEKLER[x],
    )

    secim_b = st.radio(
        "Tesadüfi eşik (B)",
        [
            "Her görsel için kendi değeri (önerilen)",
            "Sabit Penrose değeri (0,528)",
        ],
    )
    sabit = secim_b.startswith("Sabit")

    koyu_modu = st.checkbox(
        "Merkezi koyu bölgeye göre ayarla",
        value=False,
    )

    hassaslik = st.selectbox(
        "Ölçüm hassasiyeti",
        list(HASSASLIK),
        index=1,
    )
    aci_sayisi, adim = HASSASLIK[hassaslik]

    st.caption(PROTOKOL_ADI)
    st.caption(
        "Negatif kontrol her ölçümde otomatik yapılır. "
        "Görsel bazında iyi huylu/kötü huylu tahmini üretilmez."
    )


# ============================ ANA SAYFA ============================
st.title("🌀 YSÜM · Yerel Simetri Üstünlük Modeli")
st.caption("Simetri ve lezyon ölçütlerini şans beklentisine göre karşılaştıran araştırma uygulaması.")

t_kavram, t_yy, t_toplu, t_isi, t_sinir = st.tabs([
    "📖 Kavram",
    "🔍 Yan Yana",
    "🧪 Toplu Test",
    "🗺️ Isı Haritası",
    "⚠️ Sınırlılıklar",
])


# ---------------------------- KAVRAM ----------------------------
with t_kavram:
    s1, s2, s3 = st.tabs(["Penrose", "Ölçütler", "Uygulamalar"])

    with s1:
        st.markdown("""
**Penrose döşemeleri**, düzenli ama periyodik olmayan döşemelerdir.
Kalın ve ince rombusların sayı oranı altın orana yaklaşır. Modeldeki
p = 1/φ değeri bu matematiksel ilişkiden gelir.

Shechtman ve arkadaşlarının kuasikristal çalışması, periyodik olmayan
düzenli yapıların fiziksel malzemelerde de bulunabileceğini göstermiştir.
Lu ve Steinhardt ise girih desenlerindeki on katlı düzenleri incelemiştir.
""")
        c1, c2, c3 = st.columns(3)
        with c1:
            goster(fig_kirinim())
        with c2:
            goster(fig_rozet())
        with c3:
            goster(fig_girih())

    with s2:
        st.markdown("""
**A · Simetri.** Görsel gri tonlamaya çevrilir, 200 × 200 piksele getirilir
ve ortalama parlaklığa göre ikileştirilir. Şans eşiği B(p) = p² + (1 − p)²,
simetri oranı S_k ve üstünlük skoru aşağıdaki gibidir:
""")
        st.latex(r"U_k = \frac{S_k - B}{1 - B}")
        st.markdown("""
**B · Kenar düzensizliği.** Lezyon, Otsu eşiğiyle ayrılır ve merkeze en yakın
en büyük bölge alınır. Sınır uzunluğu P ve alan A ile P / (2√(πA)) hesaplanır.
Daire yaklaşık 1 verir; değer büyüdükçe sınır daha düzensizdir.

**C · Renk çeşitliliği.** Lezyondaki pikseller 6 yaklaşık dermatoskopik renge
(beyaz, açık ve koyu kahverengi, mavi-gri, siyah, kırmızımsı) en yakın olana
atanır. Piksellerin en az %5'ini alan renk sınıfları sayılır.

**Kalite kuralları.** A ölçümünde geçerli piksel çifti oranı %50'nin altındaki
noktalar ve tamamen tek renkli bölgeler hesaba girmez.

|U| ≤ 0,15 sınırı yalnızca açıklama konvansiyonudur; istatistiksel eşik
değildir.
""")

        satirlar = []
        K, I = 2, 1
        for n in range(1, 7):
            T = K + I
            satirlar.append({
                "n": n,
                "K_n": K,
                "I_n": I,
                "T_n": T,
                "K_n/T_n": round(K / T, 6),
            })
            K, I = 2 * K + I, K + I

        st.table(pd.DataFrame(satirlar))
        goster(fig_b_egrisi())

    with s3:
        st.info(
            "Dermatoskopik çalışmada nv ve mel adları metadata'daki grupları "
            "belirtir. YSÜM tek bir görsel hakkında iyi huylu veya kötü huylu "
            "kararı vermez."
        )


# ---------------------------- YAN YANA ----------------------------
with t_yy:
    st.subheader("İki görseli yan yana analiz")

    c1, c2 = st.columns(2)
    with c1:
        g1, rgb1, ad1 = gorsel_sec("1. görsel", "yy1", "Yuvarlak şekil")
    with c2:
        g2, rgb2, ad2 = gorsel_sec("2. görsel", "yy2", "Düzensiz şekil")

    if g1 is None or g2 is None:
        st.info("İki görsel seçin veya yükleyin.")
    else:
        r1 = gorsel_analiz(g1, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        r2 = gorsel_analiz(g2, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        k1 = kriter_olc(g1, rgb1)
        k2 = kriter_olc(g2, rgb2)

        if r1 is None or r2 is None:
            st.error("Görsellerden birinde yeterli geçerli ölçüm noktası bulunamadı.")
        else:
            with c1:
                st.image(g1, caption=ad1, width=280)
            with c2:
                st.image(g2, caption=ad2, width=280)

            def bicim(v):
                if isinstance(v, (bool, np.bool_)):
                    return "Evet" if v else "Hayır"
                if isinstance(v, (int, np.integer)):
                    return str(int(v))
                if isinstance(v, (float, np.floating)):
                    return tr_sayi(float(v))
                return str(v)

            def renk_sayi(kr):
                return str(int(kr["C_renk"])) if kr["Lezyon_bulundu"] else "—"

            satirlar = [
                ("A · Açık alan oranı (p)", r1["p"], r2["p"]),
                ("A · Şans eşiği (B)", r1["B"], r2["B"]),
                ("A · Ortalama eşleşme (S_ort)", r1["S_ort"], r2["S_ort"]),
                ("A · Ortalama YSÜM (U_ort)", r1["U_ort"], r2["U_ort"]),
                ("A · Medyan YSÜM", r1["U_medyan"], r2["U_medyan"]),
                ("A · Izgara içi dağılım (U_std)", r1["U_std"], r2["U_std"]),
                ("A · Maksimum odak (U_maks)", r1["U_maks"], r2["U_maks"]),
                ("A · Merkez YSÜM", r1["U_merkez"], r2["U_merkez"]),
                ("A · Negatif kontrol", r1["U_kontrol"], r2["U_kontrol"]),
                ("A · Geçerli nokta", r1["Gecerli_nokta"], r2["Gecerli_nokta"]),
                ("A · Ortalama kapsama", r1["Kapsama_ort"], r2["Kapsama_ort"]),
                ("Lezyon bulundu", k1["Lezyon_bulundu"], k2["Lezyon_bulundu"]),
                (
                    "B · Kenar düzensizliği",
                    tr_sayi(k1["B_duz"], 3) if k1["Lezyon_bulundu"] else "—",
                    tr_sayi(k2["B_duz"], 3) if k2["Lezyon_bulundu"] else "—",
                ),
                ("C · Renk sınıfı sayısı", renk_sayi(k1), renk_sayi(k2)),
                ("C · Bulunan renk sınıfları", k1["C_siniflar"] or "—", k2["C_siniflar"] or "—"),
                (
                    "Durum (A, U_ort'a göre)",
                    yorum_etiketi(r1["U_ort"]),
                    yorum_etiketi(r2["U_ort"]),
                ),
            ]

            tablo = pd.DataFrame(
                [(b, bicim(v1), bicim(v2)) for b, v1, v2 in satirlar],
                columns=["Ölçüt", "1. görsel", "2. görsel"],
            )
            st.table(tablo)

            delta = r2["U_ort"] - r1["U_ort"]
            st.metric("ΔU = U_ort(2) − U_ort(1)", f"{delta:+.4f}")

            st.caption(
                "Bu tablo yalnızca iki görselin ölçüm değerlerini gösterir. "
                "Görseller iyi huylu/kötü huylu olarak sınıflandırılmaz. "
                "Grup karşılaştırması için Toplu Test kullanılır."
            )


# ---------------------------- TOPLU TEST ----------------------------
with t_toplu:
    st.subheader("Üç ölçütle iki grup karşılaştırması")

    st.markdown(
        "Grupları metadata'ya göre yükleyin. Üç ölçüt (A, B, C) aynı görsellerle "
        "hesaplanır. Anlamlılık eşiği 0,05 / 3 ≈ 0,0167'dir."
    )

    v = st.session_state.setdefault("yukleme_v", 0)

    c1, c2 = st.columns(2)
    with c1:
        iyi_yuk = st.file_uploader(
            GRUP_IYI,
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"grp_iyi_{v}",
        )
    with c2:
        kotu_yuk = st.file_uploader(
            GRUP_KOTU,
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"grp_kotu_{v}",
        )

    b1, b2 = st.columns(2)
    calistir = b1.button("🚀 Testi çalıştır", key="btn_toplu", type="primary")
    temizle = b2.button("🗑️ Eski sonuçları ve yüklemeleri temizle", key="btn_temizle")

    if temizle:
        st.session_state.pop("toplu", None)
        st.session_state["yukleme_v"] = v + 1
        st.rerun()

    if calistir:
        iyi = [(f.name, f.getvalue()) for f in (iyi_yuk or [])]
        kotu = [(f.name, f.getvalue()) for f in (kotu_yuk or [])]

        if len(iyi) < 2 or len(kotu) < 2:
            st.error("Her grupta en az iki görsel yüklenmelidir.")
        else:
            ad_ortak = {a for a, _ in iyi} & {a for a, _ in kotu}
            md5_iyi = {hashlib.md5(b).hexdigest() for _, b in iyi}
            md5_kotu = {hashlib.md5(b).hexdigest() for _, b in kotu}
            icerik_ortak = md5_iyi & md5_kotu

            if ad_ortak or icerik_ortak:
                st.error("Aynı görsel iki grupta bulunuyor. Test durduruldu.")
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


# ---------------------------- ISI HARİTASI ----------------------------
with t_isi:
    st.subheader("Isı haritası ve odak noktaları")

    gri, _, ad = gorsel_sec("Görsel", "isi", "5 katlı desen")

    k_isi = st.selectbox(
        "Simetri türü (ısı haritası)",
        list(K_SECENEKLER),
        index=1,
        format_func=lambda x: K_SECENEKLER[x],
        key="isi_k",
    )

    if gri is not None:
        r = gorsel_analiz(
            gri, k_isi, sabit, koyu_modu,
            ISI_YARICAPLARI, False, aci_sayisi, adim,
        )

        if r is None:
            st.error("Yeterli geçerli ölçüm noktası bulunamadı.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(gri, caption=ad, width=280)
            with c2:
                goster(isi_figur(
                    gri,
                    r["izg"],
                    r["U_matris"],
                    f"k = {k_isi} · U_ort = {r['U_ort']:.3f}",
                ))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Açık alan oranı", f"{r['p']:.3f}")
            m2.metric("Ortalama YSÜM", f"{r['U_ort']:.4f}")
            m3.metric("Medyan YSÜM", f"{r['U_medyan']:.4f}")
            m4.metric(
                "Geçerli / toplam nokta",
                f"{r['Gecerli_nokta']} / {r['Toplam_nokta']}",
            )

            st.caption(
                "Boş ısı haritası noktaları kalite kuralı nedeniyle hesaplamaya "
                "alınmamıştır."
            )

            u = r["U_matris"]
            if np.isfinite(u).any():
                doldurulmus = np.where(np.isfinite(u), u, -np.inf)
                sira = np.argsort(doldurulmus.ravel())[::-1][:3]

                odaklar = []
                for idx in sira:
                    iy, ix = np.unravel_index(idx, u.shape)
                    odaklar.append({
                        "x": int(r["izg"][ix]),
                        "y": int(r["izg"][iy]),
                        "U": round(float(u[iy, ix]), 4),
                    })
                st.table(pd.DataFrame(odaklar))


# ---------------------------- SINIRLILIKLAR ----------------------------
with t_sinir:
    st.markdown("""
- Uygulama tıbbi tanı, tarama veya karar aracı değildir.
- Tek bir görsel iyi huylu/kötü huylu olarak sınıflandırılmaz.
- Görseller 200 × 200 piksele getirilir; en-boy oranı korunmaz.
- **B ve C lezyon bölütlemesi basit eşiklemeye dayanır.** Saç, kıl izi ve
  gölgeler bölütlemeyi bozabilir. Lezyon bulunamayan görseller B ve C
  analizinden çıkarılır; bu sayı tabloda gösterilir.
- **C renk sınıfları yaklaşıktır.** Palet sabit ve önceden tanımlıdır; kalibre
  edilmiş bir renk ölçümü değildir.
- 180° nokta simetrisi ile dermatolojideki eksen simetrisi aynı değildir.
- Kalite kuralları kenar ve ardıplan etkisini azaltır; bütün ölçüm hatalarını
  ortadan kaldırmaz.
- ±0,15 sınırı bir açıklama konvansiyonudur; istatistiksel test değildir.
- Üç ölçüt için Bonferroni düzeltmesi uygulanır (α = 0,05 / 3).
- Dengeli 100/100 örneklem gerçek hastalık sıklığını yansıtmaz.
- Sonuçlar örneklem, ön işleme ve ölçüm protokolüne özgüdür.
""")

    st.error(
        "Bu test klinik araştırma standartlarının altındadır. "
        "Sonuç hiçbir şekilde tıbbi geçerlilik taşımaz."
    )

st.markdown("---")
st.caption(
    "⚠️ Araştırma ve eğitim amaçlıdır. "
    "Tıbbi tanı, tarama veya karar için kullanılamaz."
)
