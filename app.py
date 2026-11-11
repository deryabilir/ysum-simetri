import hashlib
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import stats

st.set_page_config(
    page_title="YSÜM · Yerel Simetri Üstünlük Modeli",
    page_icon="🌀",
    layout="wide",
)

# ============================ SABİTLER ============================
PHI = (1 + math.sqrt(5)) / 2
P_PENROSE = 1 / PHI
B_PENROSE = P_PENROSE**2 + (1 - P_PENROSE) ** 2  # ≈ 0,528
IMG_N = 200
ALFA = 0.05
YORUM_ESIGI = 0.15
IZG_BASLANGIC = 10
GRUP_IYI = "İyi huylu (nv)"
GRUP_KOTU = "Kötü huylu (mel)"
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


# ============================ ÇİZİMLER ============================
def goster(fig):
    st.pyplot(fig)
    plt.close(fig)


def fig_kirinim():
    x = np.linspace(-1, 1, 360)
    X, Y = np.meshgrid(x, x)
    z = sum(
        np.cos(2 * np.pi * 6 * (X * np.cos(2 * np.pi * j / 5) + Y * np.sin(2 * np.pi * j / 5)))
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
    e = (izg[0], izg[-1], izg[-1], izg[0])
    im = ax.imshow(
        np.nan_to_num(u_matris), cmap="RdBu_r", vmin=-0.5, vmax=0.5, alpha=0.6, extent=e
    )
    ax.set_xlim(0, IMG_N)
    ax.set_ylim(IMG_N, 0)
    ax.set_title(baslik, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def toplu_grafik(df):
    fig, eksen = plt.subplots(1, 2, figsize=(10, 4))
    secenekler = (
        (eksen[0], "U_ort", "Gerçek görseller (U_ort)"),
        (eksen[1], "U_kontrol", "Negatif kontrol (U_kontrol)"),
    )
    for ax, kolon, baslik in secenekler:
        veri = [df.loc[df["Grup"] == g, kolon].dropna().values for g in (GRUP_IYI, GRUP_KOTU)]
        ax.boxplot(veri, widths=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["İyi huylu (nv)", "Kötü huylu (mel)"])
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(YORUM_ESIGI, color="#c0392b", ls=":", lw=0.8)
        ax.axhline(-YORUM_ESIGI, color="#c0392b", ls=":", lw=0.8)
        ax.set_title(baslik, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ============================ GÖRÜNTÜ İŞLEME ============================
def gorsel_oku(kaynak):
    """Görseli gri tonlamaya çevirir, IMG_N × IMG_N boyutuna getirir. Hata olursa None."""
    try:
        img = Image.open(kaynak).convert("L").resize((IMG_N, IMG_N), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


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
    """Koyu piksellerin ağırlık merkezi; koyu oran makul değilse görselin ortası."""
    koyu = ~binary
    if 0.02 <= koyu.mean() <= 0.98:
        ys, xs = np.nonzero(koyu)
        return int(round(xs.mean())), int(round(ys.mean()))
    return IMG_N // 2, IMG_N // 2


def sim_hesapla(binary, k, cxs, cys, radii, aci_sayisi, parca=256):
    """Her merkez noktası için S_k: eşleşen çift / geçerli çift. Parça parça hesaplanır."""
    cxs = np.asarray(cxs, dtype=float)
    cys = np.asarray(cys, dtype=float)
    r = np.asarray(radii, dtype=float)[None, :, None]
    th = np.linspace(0.0, 2 * np.pi, aci_sayisi, endpoint=False)[None, None, :]
    kayma = 2 * np.pi / k
    cos1, sin1 = np.cos(th), np.sin(th)
    cos2, sin2 = np.cos(th + kayma), np.sin(th + kayma)
    sonuc = []
    for bas in range(0, len(cxs), parca):
        cx = cxs[bas : bas + parca, None, None]
        cy = cys[bas : bas + parca, None, None]
        x1 = np.rint(cx + r * cos1).astype(int)
        y1 = np.rint(cy + r * sin1).astype(int)
        x2 = np.rint(cx + r * cos2).astype(int)
        y2 = np.rint(cy + r * sin2).astype(int)
        gecerli = (
            (x1 >= 0) & (x1 < IMG_N) & (y1 >= 0) & (y1 < IMG_N)
            & (x2 >= 0) & (x2 < IMG_N) & (y2 >= 0) & (y2 < IMG_N)
        )
        x1c, y1c = np.clip(x1, 0, IMG_N - 1), np.clip(y1, 0, IMG_N - 1)
        x2c, y2c = np.clip(x2, 0, IMG_N - 1), np.clip(y2, 0, IMG_N - 1)
        esles = (binary[y1c, x1c] == binary[y2c, x2c]) & gecerli
        toplam = gecerli.sum(axis=(1, 2))
        eslesme = esles.sum(axis=(1, 2))
        sonuc.append(np.where(toplam > 0, eslesme / np.maximum(toplam, 1), np.nan))
    return np.concatenate(sonuc)


def gorsel_analiz(gri, k, sabit, koyu_modu, radii, kontrol, aci_sayisi, adim):
    """Bir görsel için p, B, S, U ölçümlerini döndürür. Tek renkli görselde None."""
    esik = float(gri.mean())
    binary = gri > esik
    p = float(binary.mean())
    if p <= 0.0 or p >= 1.0:
        return None
    B = B_PENROSE if sabit else p**2 + (1 - p) ** 2

    cx, cy = merkez_bul(binary) if koyu_modu else (IMG_N // 2, IMG_N // 2)

    izg = np.arange(IZG_BASLANGIC, IMG_N - IZG_BASLANGIC + 1, adim)
    gx, gy = np.meshgrid(izg, izg)
    xs, ys = gx.ravel(), gy.ravel()

    s_izg = sim_hesapla(binary, k, xs, ys, radii, aci_sayisi)
    u_izg = (s_izg - B) / (1 - B)
    gecerli = np.isfinite(u_izg)
    if not gecerli.any():
        return None
    u = u_izg[gecerli]
    s = s_izg[gecerli]
    u_merkez = (sim_hesapla(binary, k, [cx], [cy], radii, aci_sayisi)[0] - B) / (1 - B)

    sonuc = {
        "p": p,
        "B": B,
        "S_ort": float(s.mean()),
        "U_ort": float(u.mean()),
        "U_std": float(u.std(ddof=1)) if u.size > 1 else 0.0,
        "U_maks": float(u.max()),
        "U_merkez": float(u_merkez),
        "izg": izg,
        "U_matris": u_izg.reshape(gx.shape),
    }
    if kontrol:
        # Negatif kontrol: pikseller karıştırılır, aynı eşik ve aynı merkezler kullanılır
        karisik = np.random.default_rng(0).permutation(gri.ravel()).reshape(gri.shape)
        k_bin = karisik > esik
        s_k = sim_hesapla(k_bin, k, xs, ys, radii, aci_sayisi)
        u_k = (s_k - B) / (1 - B)
        sonuc["U_kontrol"] = float(np.nanmean(u_k))
    return sonuc


def yorum_etiketi(u):
    if u is None or not np.isfinite(u):
        return "Hesaplanamadı"
    if u > YORUM_ESIGI:
        return "Tesadüften belirgin yüksek"
    if u < -YORUM_ESIGI:
        return "Tesadüften belirgin düşük"
    return "Tesadüf beklentisine yakın"


# ============================ İSTATİSTİK ============================
def iki_grup_istatistik(a, b):
    """Tek yönlü Welch t-testi (H1: μ_a > μ_b), Cohen's d ve %95 güven aralığı."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float)
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return None
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    fark = a.mean() - b.mean()
    tk = stats.t.ppf(1 - ALFA / 2, df)
    se = math.sqrt(se2)
    return {
        "n1": n1,
        "n2": n2,
        "fark": float(fark),
        "t": float(t),
        "p": float(p),
        "d": float(fark / sp) if sp > 0 else float("nan"),
        "ci": (float(fark - tk * se), float(fark + tk * se)),
    }


def toplu_calistir(dosyalar, k, sabit, koyu_modu, aci_sayisi, adim):
    """dosyalar: [(grup, ad, bayt), ...]"""
    satirlar, okunamayan = [], []
    toplam = len(dosyalar)
    bar = st.progress(0.0, text="Görseller analiz ediliyor...")
    for i, (grup, ad, veri) in enumerate(dosyalar):
        gri = gorsel_oku(io.BytesIO(veri))
        r = None
        if gri is not None:
            r = gorsel_analiz(gri, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        if r is None:
            okunamayan.append(ad)
        else:
            satirlar.append(
                {
                    "Grup": grup,
                    "Dosya": ad,
                    "p": r["p"],
                    "B": r["B"],
                    "S_ort": r["S_ort"],
                    "U_ort": r["U_ort"],
                    "U_std": r["U_std"],
                    "U_maks": r["U_maks"],
                    "U_kontrol": r["U_kontrol"],
                }
            )
        bar.progress((i + 1) / toplam, text=f"{i + 1}/{toplam} görsel")
    bar.empty()
    return pd.DataFrame(satirlar), okunamayan


def toplu_goster(s):
    df = s["df"]
    st.caption(
        f"Bu sonuç şu ayarlarla üretildi: k = {s['k']} · B: {s['b']} · hassasiyet: {s['hassasiyet']}"
    )
    if df.empty:
        st.error("Hiçbir görsel analiz edilemedi.")
        return

    iyi = df.loc[df["Grup"] == GRUP_IYI]
    kotu = df.loc[df["Grup"] == GRUP_KOTU]

    ozet = (
        df.groupby("Grup")
        .agg(
            gorsel=("Dosya", "count"),
            s_ort=("S_ort", "mean"),
            u_ort=("U_ort", "mean"),
            u_sd=("U_ort", "std"),
            u_kontrol=("U_kontrol", "mean"),
            kararlilik=("U_std", "mean"),
        )
        .reindex([GRUP_IYI, GRUP_KOTU])
        .rename(
            columns={
                "gorsel": "Görsel sayısı",
                "s_ort": "Ortalama S",
                "u_ort": "Ortalama U_ort",
                "u_sd": "U_ort std. sapma",
                "u_kontrol": "Ortalama U_kontrol",
                "kararlilik": "Ortalama ızgara sapması (U_std)",
            }
        )
    )
    st.markdown("#### Grup özeti")
    st.dataframe(ozet.round(4))

    sonuc = iki_grup_istatistik(iyi["U_ort"], kotu["U_ort"])
    if sonuc is None:
        st.warning("İstatistik için her grupta en az 2 geçerli görsel gerekir.")
    else:
        m = st.columns(5)
        m[0].metric("ΔU (iyi − kötü)", f"{sonuc['fark']:+.4f}")
        m[1].metric("Welch t", f"{sonuc['t']:.3f}")
        m[2].metric("p (tek yönlü)", f"{sonuc['p']:.4f}")
        m[3].metric("Cohen's d", f"{sonuc['d']:.3f}")
        lo, hi = sonuc["ci"]
        m[4].metric("%95 güven aralığı (fark)", f"[{lo:+.4f}, {hi:+.4f}]")
        if sonuc["p"] < ALFA:
            st.success("Tek yönlü test α = 0,05 düzeyinde anlamlı bir fark gösterdi.")
        else:
            st.info(
                "Tek yönlü test α = 0,05 düzeyinde anlamlı bir fark göstermedi. "
                "Bu da bir sonuçtur ve olduğu gibi raporlanır."
            )

    if len(iyi) >= 2 and len(kotu) >= 2:
        goster(toplu_grafik(df))

    st.markdown("#### Görsel bazlı sonuçlar")
    st.dataframe(df.round(4))

    if s["okunamayan"]:
        st.warning("Analiz edilemeyen dosyalar: " + ", ".join(s["okunamayan"]))

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ysum_sonuclar.csv indir",
        csv,
        file_name="ysum_sonuclar.csv",
        mime="text/csv",
    )


# ============================ GİRDİ YARDIMCILARI ============================
def gorsel_sec(etiket, anahtar, varsayilan=None):
    """Yükleme varsa yüklenen görsel, yoksa seçilen örnek kullanılır."""
    yuklenen = st.file_uploader(f"{etiket} (yükle)", type=UPLOAD_TYPES, key=f"{anahtar}_up")
    secenekler = ["(yok)"] + list(ORNEKLER)
    idx = secenekler.index(varsayilan) if varsayilan in secenekler else 0
    ornek = st.selectbox(f"{etiket} (örnek)", secenekler, index=idx, key=f"{anahtar}_ex")
    if yuklenen is not None:
        gri = gorsel_oku(io.BytesIO(yuklenen.getvalue()))
        if gri is None:
            st.error("Bu dosya okunamadı.")
            return None, None
        return gri, yuklenen.name
    if ornek != "(yok)":
        return ornek_uret(ORNEKLER[ornek]), ornek
    return None, None


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
        ["Her görsel için kendi değeri (önerilen)", "Sabit Penrose değeri (0,528)"],
    )
    sabit = secim_b.startswith("Sabit")
    koyu_modu = st.checkbox("Merkezi koyu bölgeye göre ayarla", value=False)
    hassaslik = st.selectbox("Ölçüm hassasiyeti", list(HASSASLIK), index=1)
    aci_sayisi, adim = HASSASLIK[hassaslik]
    st.caption("Negatif kontrol (piksel karıştırılmış görsel) her ölçümde otomatik yapılır.")

# ============================ ANA SAYFA ============================
st.title("🌀 YSÜM · Yerel Simetri Üstünlük Modeli")
st.caption("Simetri ölçümünü şans beklentisine göre düzelten bir araştırma uygulaması.")

t_kavram, t_yy, t_toplu, t_isi, t_sinir = st.tabs(
    ["📖 Kavram", "🔍 Yan Yana", "🧪 Toplu Test", "🗺️ Isı Haritası", "⚠️ Sınırlılıklar"]
)

# ---------------------------- KAVRAM ----------------------------
with t_kavram:
    s1, s2, s3 = st.tabs(["Penrose", "Ölçüm", "Uygulamalar"])

    with s1:
        st.markdown(
            """
**Penrose döşemeleri**, düzenli ama periyodik olmayan (aperiyodik) döşemelerdir.
Roger Penrose (1974), kalın ve ince iki rombus ile kurulan, hiçbir yerde tam olarak
tekrarlanmayan bir döşeme ailesi tanımladı. Kalın ve ince rombusların sayı oranı altın
orana (φ ≈ 1,618) yaklaşır. Modelde şans eşiğinin dayandığı p = 1/φ değeri buradan gelir.

**Kuasikristaller.** Shechtman ve ark. (1984), periyodik olmayan ama düzenli açısal
simetri gösteren bir alaşım gözlemledi. Bu keşif 2011 Kimya Nobel Ödülü ile
ödüllendirildi (Kraliyet İsveç Bilimler Akademisi, 2011).

**Girih desenleri.** Lu ve Steinhardt (2007), ortaçağ İslam mimarisindeki girih
desenlerinin on katlı, Penrose benzeri düzenler içerdiğini bildirdi. Bu bir kaynak
bulgusudur; bu projenin bulgusu değildir.
"""
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            goster(fig_kirinim())
        with c2:
            goster(fig_rozet())
        with c3:
            goster(fig_girih())

    with s2:
        st.markdown(
            """
**Adım 1 · Şans eşiği.** Görsel gri tonlamaya çevrilir, 200 × 200 piksele getirilir ve
ortalama parlaklığa göre açık/koyu olarak ikileştirilir. Açık piksel oranı p ise şans
eşiği B(p) = p² + (1 − p)²'dir.

**Adım 2 · Simetri ölçümü.** Merkezden r yarıçapında bir θ noktası ile θ + 360°/k noktası
karşılaştırılır. Eşleşen çiftlerin oranı S_k'dır.

**Adım 3 · Üstünlük skoru.**
"""
        )
        st.latex(r"U_k = \frac{S_k - B}{1 - B}")
        st.markdown(
            """
**Adım 4 · Yorum.** |U| ≤ 0,15 tesadüf beklentisine yakındır. U > 0,15 tesadüften
belirgin yüksek, U < −0,15 belirgin düşüktür. Bu sınır bir konvansiyondur, istatistiksel
eşik değildir.

**Adım 5 · Negatif kontrol.** Aynı görselin pikselleri rastgele karıştırılır ve ölçüm
tekrarlanır. Karıştırılmış görselin skoru (U_kontrol) ayrı gösterilir.
"""
        )
        st.markdown("**Tablo 1.** Dönüşüm dizisi ve K_n/T_n oranı")
        satirlar = []
        K, I = 2, 1
        for n in range(1, 7):
            T = K + I
            satirlar.append({"n": n, "K_n": K, "I_n": I, "T_n": T, "K_n/T_n": round(K / T, 6)})
            K, I = 2 * K + I, K + I
        st.table(pd.DataFrame(satirlar))
        goster(fig_b_egrisi())

    with s3:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("#### Dermatoskopi ile ilişki")
            st.markdown(
                "ABCDE kuralının A (asimetri) kriteri eksen simetrisine dayanır. Bu uygulama "
                "180° nokta simetrisini ölçer; k = 2 bu ilişki için bir vekil ölçüdür. "
                "Tanı aracı değildir."
            )
        with k2:
            st.markdown("#### Desen analizi")
            st.markdown(
                "Tarihî geometrik desenlerde baskın dönel simetri derecesini (k) ve "
                "YSÜM değerini belirlemek için kullanılabilir."
            )
        with k3:
            st.markdown("#### Görüntü karşılaştırma")
            st.markdown(
                "İki görselin simetri üstünlüklerini aynı ölçüte göre yan yana karşılaştırır "
                "ve grup düzeyinde test eder."
            )

# ---------------------------- YAN YANA ----------------------------
with t_yy:
    st.subheader("İki görseli yan yana analiz")
    c1, c2 = st.columns(2)
    with c1:
        g1, ad1 = gorsel_sec("1. görsel", "yy1", "Yuvarlak şekil")
    with c2:
        g2, ad2 = gorsel_sec("2. görsel", "yy2", "Düzensiz şekil")

    if g1 is None or g2 is None:
        st.info("İki görsel seçin: yükleyin ya da örnek seçin.")
    else:
        r1 = gorsel_analiz(g1, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        r2 = gorsel_analiz(g2, k, sabit, koyu_modu, SKOR_YARICAPLARI, True, aci_sayisi, adim)
        if r1 is None or r2 is None:
            st.error("Tek renkli (düz) görsel ölçülemez. Başka bir görsel seçin.")
        else:
            with c1:
                st.image(g1, caption=ad1, width=280)
            with c2:
                st.image(g2, caption=ad2, width=280)

            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else str(v)

            satirlar = [
                ("Açık alan oranı (p)", r1["p"], r2["p"]),
                ("Şans eşiği (B)", r1["B"], r2["B"]),
                ("Ortalama eşleşme (S_ort)", r1["S_ort"], r2["S_ort"]),
                ("Ortalama YSÜM (U_ort)", r1["U_ort"], r2["U_ort"]),
                ("Ölçüm kararlılığı (U_std)", r1["U_std"], r2["U_std"]),
                ("Maksimum odak (U_maks)", r1["U_maks"], r2["U_maks"]),
                ("Merkez YSÜM (U_merkez)", r1["U_merkez"], r2["U_merkez"]),
                ("Negatif kontrol (U_kontrol)", r1["U_kontrol"], r2["U_kontrol"]),
                ("Durum (U_ort'a göre)", yorum_etiketi(r1["U_ort"]), yorum_etiketi(r2["U_ort"])),
            ]
            tablo = pd.DataFrame(
                [(a, fmt(b), fmt(c)) for a, b, c in satirlar],
                columns=["Ölçüt", "1. görsel", "2. görsel"],
            )
            st.table(tablo)
            delta = r2["U_ort"] - r1["U_ort"]
            st.metric("ΔU = U_ort(2) − U_ort(1)", f"{delta:+.4f}")
            st.caption(
                "ΔU pozitifse 2. görsel, 1. göre daha yüksek simetri üstünlüğü gösterir. "
                "Tek bir görsel çifti genelleme yapmaya yetmez; grup düzeyi için Toplu Test kullanın."
            )

# ---------------------------- TOPLU TEST ----------------------------
with t_toplu:
    st.subheader("İki grup karşılaştırma testi")
    st.markdown(
        "Her grupta en az 2 görsel yükleyin. Rapordaki test **k = 2** ile yapılmıştır. "
        "Aynı görsel iki gruba birden yüklenemez; bu kontrol dosya adı ve içerik (md5) "
        "üzerinden yapılır."
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
            st.error("Her grupta en az 2 görsel yüklenmelidir.")
        else:
            ad_ortak = {a for a, _ in iyi} & {a for a, _ in kotu}
            md5_iyi = {hashlib.md5(b).hexdigest() for _, b in iyi}
            md5_kotu = {hashlib.md5(b).hexdigest() for _, b in kotu}
            md5_ortak = md5_iyi & md5_kotu
            if ad_ortak or md5_ortak:
                st.error("Aynı görsel iki grupta birden var. Test durduruldu; lütfen düzeltin.")
                if ad_ortak:
                    st.write("Aynı dosya adları: " + ", ".join(sorted(ad_ortak)))
                if md5_ortak:
                    st.write(f"Aynı içerikli dosya sayısı: {len(md5_ortak)}")
                st.session_state.pop("toplu", None)
            else:
                dosyalar = [(GRUP_IYI, a, b) for a, b in iyi] + [(GRUP_KOTU, a, b) for a, b in kotu]
                df, okunamayan = toplu_calistir(dosyalar, k, sabit, koyu_modu, aci_sayisi, adim)
                st.session_state["toplu"] = {
                    "df": df,
                    "okunamayan": okunamayan,
                    "k": k,
                    "b": "Sabit 0,528" if sabit else "Her görsel için kendi değeri",
                    "hassasiyet": hassaslik,
                }

    if "toplu" in st.session_state:
        toplu_goster(st.session_state["toplu"])

# ---------------------------- ISI HARİTASI ----------------------------
with t_isi:
    st.subheader("Isı haritası ve odak noktaları")
    gri, ad = gorsel_sec("Görsel", "isi", "5 katlı desen")
    k_isi = st.selectbox(
        "Simetri türü (ısı haritası)",
        list(K_SECENEKLER),
        index=1,
        format_func=lambda x: K_SECENEKLER[x],
        key="isi_k",
    )
    if gri is None:
        st.info("Bir görsel yükleyin ya da örnek seçin.")
    else:
        r = gorsel_analiz(gri, k_isi, sabit, koyu_modu, ISI_YARICAPLARI, False, aci_sayisi, adim)
        if r is None:
            st.error("Tek renkli (düz) görsel ölçülemez.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(gri, caption=ad, width=280)
            with c2:
                goster(isi_figur(gri, r["izg"], r["U_matris"], f"k = {k_isi} · U_ort = {r['U_ort']:.3f}"))
            m1, m2, m3 = st.columns(3)
            m1.metric("Açık alan oranı", f"{r['p']:.3f}")
            m2.metric("Merkez YSÜM", f"{r['U_merkez']:.4f}")
            m3.metric("Ortalama YSÜM", f"{r['U_ort']:.4f}")

            st.markdown("**En yüksek üç odak noktası** (merkez koordinatı, piksel)")
            u = np.nan_to_num(r["U_matris"], nan=-np.inf)
            sira = np.argsort(u.ravel())[::-1][:3]
            odaklar = []
            for idx in sira:
                iy, ix = np.unravel_index(idx, u.shape)
                odaklar.append(
                    {
                        "x": int(r["izg"][ix]),
                        "y": int(r["izg"][iy]),
                        "U": round(float(r["U_matris"][iy, ix]), 4),
                    }
                )
            st.table(pd.DataFrame(odaklar))

# ---------------------------- SINIRLILIKLAR ----------------------------
with t_sinir:
    st.markdown(
        """
- **Tıbbi değildir.** Uygulama tanı, tarama veya karar için kullanılamaz.
- **Ön işleme.** Görseller gri tonlamaya ve 200 × 200 piksele çevrilir; en-boy oranı korunmaz.
- **Segmentasyon yoktur.** Lezyon sınırı çıkarılmaz; ölçüm tüm yüzey üzerinden yapılır.
- **Saç ve kıl izleri** ölçümü etkileyebilir.
- **Merkez seçimi** varsayılan olarak görselin ortasıdır; koyu bölgeye göre seçim isteğe bağlıdır.
- **Negatif kontrol** istatistiksel test değildir; bir karşılaştırma referansıdır.
- **Simetri türü:** 180° nokta simetrisi ile eksen simetrisi aynı değildir. k = 2 bir vekil ölçüdür.
- **Hassasiyet:** 200 × 200 piksel çözünürlüğü ölçümün üst sınırını belirler. Açı ve ızgara sıklığı
  artırıldığında ölçüm kararlılığı artar, görüntüdeki bilgi artmaz.
- **Eşikler:** ±0,15 yorum sınırı bir konvansiyondur, istatistiksel eşik değildir.
- **Örneklem:** Sonuçlar örneklem ve protokole özgüdür. Dengeli 100/100 örneklem gerçek
  sıklığı yansıtmaz. Tek merkez ve tek protokol sınırlılığı vardır.
- **HAM10000** CC BY-NC 4.0 lisansı ile yalnızca ticari olmayan kullanıma açıktır.
- **Etik:** Etik kurul görüşü danışmandan alınmalıdır.
"""
    )
    st.error(
        "Bu test klinik araştırma standartlarının altındadır. "
        "Sonuç hiçbir şekilde tıbbi geçerlilik taşımaz."
    )

st.markdown("---")
st.caption("⚠️ Araştırma ve eğitim amaçlıdır. Tıbbi tanı, tarama veya karar için kullanılamaz.")
