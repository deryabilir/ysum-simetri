import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from scipy import stats
import streamlit as st

# ==============================================================================
# SAYFA AYARI (ilk Streamlit çağrısı olmalı)
# ==============================================================================
st.set_page_config(
    page_title="YSÜM — Yapısal Simetri Laboratuvarı",
    page_icon="🌀",
    layout="wide",
)

# ==============================================================================
# SABİTLER
# ==============================================================================
PHI = (1 + 5 ** 0.5) / 2
P_PENROSE = 1 / PHI
B_PENROSE = P_PENROSE ** 2 + (1 - P_PENROSE) ** 2     # ≈ 0.528

K_SECENEKLER = {
    "2 kat · 180° (dermatoskopi)": 2,
    "5 kat · 72° (Penrose)": 5,
    "6 kat · 60°": 6,
    "8 kat · 45°": 8,
}
IMG_N = 200
RADII = (20, 40, 60, 80)
N_ACI = 36
HEAT_RADII = (10, 20, 30, 40)
HEAT_ADIM = 8
UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp"]

ORNEK_SIMETRIK = "Örnek · Yuvarlak şekil"
ORNEK_ASIMETRIK = "Örnek · Düzensiz şekil"
ORNEK_5KAT = "Örnek · 5 katlı desen"
ORNEK_GURULTU = "Örnek · Rastgele gürültü"
ORNEKLER = [ORNEK_SIMETRIK, ORNEK_ASIMETRIK, ORNEK_5KAT, ORNEK_GURULTU]

# ==============================================================================
# STİL
# ==============================================================================
st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #0b1020 0%, #1e1b4b 55%, #0e7490 100%);
       padding: 26px 30px; border-radius: 18px; color: #f8fafc;
       box-shadow: 0 8px 30px rgba(0,0,0,.25); margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .92; line-height: 1.5;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 999px; margin: 10px 8px 0 0;
        font-size: .82rem; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);}
.kart {border-radius: 14px; padding: 14px 18px; margin: 10px 0; border-left: 5px solid;
       background: rgba(127,127,127,.07);}
.kart .etiket {font-size: .70rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; opacity: .85;}
.kart h4 {margin: 4px 0 6px 0;}
.kart-ok {border-color: #22d3ee;}
.kart-prop {border-color: #fbbf24;}
.kart-warn {border-color: #ef4444;}
</style>
""", unsafe_allow_html=True)

ETIKETLER = {
    "ok": ("Bilgi", "kart-ok"),
    "prop": ("Öneri · doğrulanmadı", "kart-prop"),
    "warn": ("Sınırlılık", "kart-warn"),
}


def kart(tur, baslik, metin):
    etiket, sinif = ETIKETLER[tur]
    st.markdown(
        f'<div class="kart {sinif}"><div class="etiket">{etiket}</div>'
        f'<h4>{baslik}</h4><div>{metin}</div></div>',
        unsafe_allow_html=True,
    )


def goster(fig):
    st.pyplot(fig)
    plt.close(fig)


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


# ==============================================================================
# HERO
# ==============================================================================
st.markdown("""
<div class="hero">
  <h1>🌀 YSÜM — Yapısal Simetri Analiz Laboratuvarı</h1>
  <p>Penrose döşemelerinden esinlenen simetri ölçümüyle lezyonların simetrisini karşılaştırın.</p>
  <span class="badge">Penrose referansı B = 0,528</span>
  <span class="badge">Simetri türleri: 2 · 5 · 6 · 8 kat</span>
  <span class="badge">Araştırma amaçlıdır · tanı aracı değildir</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# YAN MENÜ
# ==============================================================================
st.sidebar.header("⚙️ Ayarlar")

k_etiket = st.sidebar.selectbox("Simetri türü", list(K_SECENEKLER.keys()), index=0)
K_SEC = K_SECENEKLER[k_etiket]

b_modu = st.sidebar.radio(
    "Tesadüfi eşik",
    ["Her görsel için kendi değeri (önerilen)", "Sabit Penrose değeri (0,528)"],
    index=0,
)
B_GORSEL = b_modu.startswith("Her")

kaydir = st.sidebar.checkbox(
    "Merkezi koyu bölgeye göre ayarla",
    value=False,
    help="Kapalıysa ölçüm görselin ortasından yapılır.",
)

# ==============================================================================
# MATEMATİK ÇEKİRDEĞİ
# ==============================================================================
def gri_yukle(dosya):
    try:
        dosya.seek(0)
    except AttributeError:
        pass
    img = Image.open(dosya).convert("L").resize((IMG_N, IMG_N), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def gri_oku_guvenli(dosya):
    try:
        return gri_yukle(dosya), None
    except Exception as e:
        return None, f"okunamadı ({type(e).__name__})"


def ornek_desen(tip):
    yy, xx = np.mgrid[0:IMG_N, 0:IMG_N]
    dx, dy = xx - 100.0, yy - 100.0
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    if tip == ORNEK_SIMETRIK:
        g = np.where(r < 55, 200, 40)
    elif tip == ORNEK_ASIMETRIK:
        sinir = 45 + 30 * np.cos(th - 0.8)
        g = np.where(r < sinir, 200, 40)
    elif tip == ORNEK_5KAT:
        f = np.cos(5 * th) + 0.5 * np.cos(10 * th) + 0.3 * np.sin(0.15 * r)
        g = np.where(f > 0, 220, 30)
    else:
        g = np.random.default_rng(42).integers(0, 256, size=(IMG_N, IMG_N))
    return np.asarray(g, dtype=np.uint8)


def ikili_matris(gri):
    esik = gri.mean()
    ikili = (gri > esik).astype(np.uint8)
    return ikili, float(ikili.mean())


def chance_B(p, b_gorsel=True):
    if not b_gorsel:
        return B_PENROSE
    if p <= 0.0 or p >= 1.0:
        return None
    return p ** 2 + (1 - p) ** 2


def u_skor(S, B):
    S = np.asarray(S, dtype=float)
    if B is None:
        return S * np.nan
    return (S - B) / (1.0 - B)


def simetri_ornekle(ikili, cxs, cys, k, radii=RADII, n_aci=N_ACI):
    cxs = np.asarray(cxs, dtype=float).ravel()
    cys = np.asarray(cys, dtype=float).ravel()
    aci = np.linspace(0.0, 2 * np.pi, n_aci, endpoint=False)
    rr = np.asarray(radii, dtype=float).reshape(1, -1, 1)
    faz = 2 * np.pi / k

    def noktalar(a):
        x = cxs.reshape(-1, 1, 1) + rr * np.cos(a).reshape(1, 1, -1)
        y = cys.reshape(-1, 1, 1) + rr * np.sin(a).reshape(1, 1, -1)
        return np.rint(x).astype(int), np.rint(y).astype(int)

    x1, y1 = noktalar(aci)
    x2, y2 = noktalar(aci + faz)
    h, w = ikili.shape
    gecerli = ((x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h) &
               (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h))
    a = ikili[np.clip(y1, 0, h - 1), np.clip(x1, 0, w - 1)]
    b = ikili[np.clip(y2, 0, h - 1), np.clip(x2, 0, w - 1)]
    eslesen = np.sum((a == b) & gecerli, axis=(1, 2))
    toplam = np.sum(gecerli, axis=(1, 2))
    return np.where(toplam > 0, eslesen / np.maximum(toplam, 1), np.nan)


def merkez_sec(ikili, kaydir_on):
    h, w = ikili.shape
    if kaydir_on:
        koyu = ikili == 0
        oran = koyu.mean()
        if 0.02 < oran < 0.98:
            ys, xs = np.nonzero(koyu)
            return float(xs.mean()), float(ys.mean())
    return w / 2.0, h / 2.0


def isi_haritasi_hesapla(ikili, B, k):
    h, w = ikili.shape
    xs = np.arange(10, w - 10, HEAT_ADIM)
    ys = np.arange(10, h - 10, HEAT_ADIM)
    GX, GY = np.meshgrid(xs, ys)
    S = simetri_ornekle(ikili, GX.ravel(), GY.ravel(), k, radii=HEAT_RADII).reshape(GX.shape)
    return u_skor(S, B), xs, ys


def analiz_et(gri, k, kaydir_on, b_gorsel=True, merkez=None, heatmap=True):
    ikili, p = ikili_matris(gri)
    B = chance_B(p, b_gorsel)
    if merkez is None:
        cx, cy = merkez_sec(ikili, kaydir_on)
    else:
        cx, cy = merkez
    S_m = float(simetri_ornekle(ikili, [cx], [cy], k)[0])
    U_m = float(u_skor(S_m, B))
    sonuc = {
        "p": p, "B": B, "cx": cx, "cy": cy,
        "S_merkez": S_m, "U_merkez": U_m,
        "U_izgara": None, "xs": None, "ys": None,
        "U_ort": float("nan"), "U_maks": float("nan"),
    }
    if heatmap:
        U_izg, xs, ys = isi_haritasi_hesapla(ikili, B, k)
        sonlu = np.isfinite(U_izg)
        sonuc.update({
            "U_izgara": U_izg, "xs": xs, "ys": ys,
            "U_ort": float(U_izg[sonlu].mean()) if sonlu.any() else float("nan"),
            "U_maks": float(U_izg[sonlu].max()) if sonlu.any() else float("nan"),
        })
    return sonuc


def negatif_kontrol_u(gri, k, kaydir_on, b_gorsel=True):
    """Pikselleri rastgele karıştırır; merkez orijinal görselden alınır."""
    ikili_orijinal, _ = ikili_matris(gri)
    merkez = merkez_sec(ikili_orijinal, kaydir_on)
    rng = np.random.default_rng(0)
    karisik = rng.permutation(gri.ravel()).reshape(gri.shape)
    return analiz_et(karisik, k, kaydir_on, b_gorsel, merkez=merkez)["U_ort"]


def duzen_yorumu(U):
    if not np.isfinite(U):
        return "Hesaplanamadı"
    if U >= 0.15:
        return "Tesadüften daha simetrik"
    if U <= -0.15:
        return "Tesadüften daha düzensiz"
    return "Tesadüfe yakın"


# ==============================================================================
# ÇİZİMLER (illüstrasyon)
# ==============================================================================
def ciz_rozet():
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=110)
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    renkler = ["#22d3ee", "#f43f8e", "#fbbf24"]
    for olcek, renk in zip([1.0, 0.66, 0.34], renkler):
        aci = np.arange(10) * np.pi / 5
        yaricap = np.where(np.arange(10) % 2 == 0, olcek, olcek * 0.382)
        pts = np.c_[yaricap * np.cos(aci), yaricap * np.sin(aci)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=renk, alpha=0.30,
                             edgecolor=renk, lw=1.6))
    ax.scatter([0], [0], c="white", s=20, zorder=5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("5 katlı desen", color="white", fontsize=9)
    return fig


def ciz_kirinim():
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=110)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    for r in [0.25, 0.45, 0.70, 0.95]:
        a1 = np.arange(5) * 2 * np.pi / 5
        a2 = a1 + np.pi / 5
        ax.scatter(r * np.cos(a1), r * np.sin(a1), c="#00f2fe", s=60 * (1.2 - r), alpha=0.9)
        ax.scatter(r * np.cos(a2), r * np.sin(a2), c="#a5f3fc", s=40 * (1.2 - r), alpha=0.7)
    ax.scatter([0], [0], c="white", s=30)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Kuasikristal kırınım deseni", color="white", fontsize=9)
    return fig


def ciz_girih():
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=110)
    aci = np.arange(10) * 2 * np.pi / 10 + np.pi / 2
    nokta = np.c_[np.cos(aci), np.sin(aci)]
    yol = [(3 * j) % 10 for j in range(10)]
    ax.add_patch(Polygon(nokta[yol], closed=True, facecolor="#14b8a6", alpha=0.30,
                         edgecolor="#0f766e", lw=2))
    ax.add_patch(Polygon(nokta, closed=True, fill=False, edgecolor="#0f766e", lw=1, ls="--"))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("10 katlı girih yıldızı", fontsize=9)
    return fig


def ciz_b_egrisi():
    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=110)
    p = np.linspace(0, 1, 200)
    ax.plot(p, p ** 2 + (1 - p) ** 2, color="#2563eb", lw=2, label="Tesadüfi eşik eğrisi")
    ax.axvline(P_PENROSE, color="#fbbf24", ls="--", label="Penrose oranı (0,618)")
    ax.axhline(B_PENROSE, color="#f43f8e", ls="--", label="Penrose eşiği (0,528)")
    ax.scatter([P_PENROSE], [B_PENROSE], color="#f43f8e", zorder=5)
    ax.set_xlabel("Açık alan oranı", fontsize=8)
    ax.set_ylabel("Tesadüfi eşik", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center")
    fig.tight_layout()
    return fig


def ciz_abcde():
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.6), dpi=110)
    th = np.linspace(0, 2 * np.pi, 300)
    r_sim = 1.0 + 0.03 * np.cos(5 * th)
    r_asim = 1.0 + 0.25 * np.sin(3 * th) + 0.10 * np.cos(7 * th)
    for ax, r, renk, baslik in [
        (axs[0], r_sim, "#16a34a", "Simetrik (örnek)"),
        (axs[1], r_asim, "#dc2626", "Asimetrik (örnek)"),
    ]:
        ax.fill(r * np.cos(th), r * np.sin(th), color=renk, alpha=0.25)
        ax.plot(r * np.cos(th), r * np.sin(th), color=renk, lw=1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(baslik, fontsize=8)
    fig.tight_layout()
    return fig


def isi_figuru(gri, sonuc, k, baslik):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
    ax.imshow(gri, cmap="gray", alpha=0.45)
    xs, ys = sonuc["xs"], sonuc["ys"]
    U = np.ma.masked_invalid(sonuc["U_izgara"])
    a = HEAT_ADIM
    im = ax.imshow(
        U, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0, alpha=0.80,
        extent=[xs[0] - a / 2, xs[-1] + a / 2, ys[-1] + a / 2, ys[0] - a / 2],
        interpolation="nearest",
    )
    ax.set_xlim(0, IMG_N)
    ax.set_ylim(IMG_N, 0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Simetri puanı", fontsize=8)
    ax.set_title(f"{baslik}\nSimetri türü: {k} kat", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ==============================================================================
# GİRDİ SEÇİCİ
# ==============================================================================
def kaynak_sec(anahtar, baslik):
    dosya = st.file_uploader(baslik, type=UPLOAD_TYPES, key=f"up_{anahtar}")
    ornek = st.selectbox("Ya da örnek seçin", ORNEKLER, key=f"ornek_{anahtar}")
    if dosya is not None:
        gri, hata = gri_oku_guvenli(dosya)
        if gri is None:
            st.error(f"{dosya.name} {hata}. Örnek görsel kullanılıyor.")
            return ornek_desen(ornek), ornek
        return gri, dosya.name
    return ornek_desen(ornek), ornek


# ==============================================================================
# TOPLU TEST YARDIMCILARI
# ==============================================================================
def toplu_calistir(dos_nv, dos_mel, kaydir_on, b_gorsel):
    kayitlar = []
    for grup, dosyalar in (("nv", dos_nv), ("mel", dos_mel)):
        for d in dosyalar:
            gri, hata = gri_oku_guvenli(d)
            if gri is None:
                kayitlar.append({
                    "Dosya": d.name, "Grup": grup, "p": np.nan, "B": np.nan,
                    "S2": np.nan, "U2": np.nan, "Hata": hata,
                })
                continue
            s = analiz_et(gri, 2, kaydir_on, b_gorsel, heatmap=False)
            kayitlar.append({
                "Dosya": d.name,
                "Grup": grup,
                "p": s["p"],
                "B": s["B"] if s["B"] is not None else np.nan,
                "S2": s["S_merkez"],
                "U2": s["U_merkez"],
                "Hata": "",
            })
    return {"df": pd.DataFrame(kayitlar), "kaydir": kaydir_on, "b_gorsel": b_gorsel}


def grup_ozeti(df_grup):
    s = df_grup["S2"].dropna()
    u = df_grup["U2"].dropna()
    return {
        "Görsel sayısı": int(len(u)),
        "Ortalama simetri": s.mean() if len(s) else np.nan,
        "En düşük": s.min() if len(s) else np.nan,
        "En yüksek": s.max() if len(s) else np.nan,
        "Ortalama puan": u.mean() if len(u) else np.nan,
    }


def toplu_grafik(nv_u, mel_u):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
    axs[0].boxplot([nv_u, mel_u], showmeans=True)
    axs[0].set_xticks([1, 2])
    axs[0].set_xticklabels(["İyi huylu", "Kötü huylu"])
    axs[0].axhline(0, color="gray", ls=":", lw=1)
    axs[0].set_ylabel("Simetri puanı")
    axs[0].set_title("Dağılım", fontsize=9)
    axs[1].hist(nv_u, bins=15, alpha=0.6, label="İyi huylu", color="#22d3ee")
    axs[1].hist(mel_u, bins=15, alpha=0.6, label="Kötü huylu", color="#f43f8e")
    axs[1].axvline(0, color="gray", ls=":", lw=1)
    axs[1].set_xlabel("Simetri puanı")
    axs[1].set_title("Dağılım (histogram)", fontsize=9)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def toplu_sonuc_goster(kayit):
    df = kayit["df"]
    st.caption(
        f"Ayarlar: {'koyu bölgeye göre merkez' if kayit['kaydir'] else 'görsel ortası'} · "
        f"{'her görsel için kendi eşiği' if kayit['b_gorsel'] else 'sabit Penrose eşiği'}"
    )

    hatali = df[df["Hata"] != ""]
    if len(hatali):
        st.warning(f"{len(hatali)} dosya okunamadı ve hesaba katılmadı.")
        with st.expander("Okunamayan dosyalar"):
            st.dataframe(hatali[["Dosya", "Grup", "Hata"]], hide_index=True)

    gecerli = df.dropna(subset=["U2"])
    nv = gecerli[gecerli["Grup"] == "nv"]
    mel = gecerli[gecerli["Grup"] == "mel"]

    ozet = pd.DataFrame([
        {"Grup": "İyi huylu", **grup_ozeti(nv)},
        {"Grup": "Kötü huylu", **grup_ozeti(mel)},
    ])
    st.table(ozet.round(3))

    nv_u = nv["U2"].to_numpy(dtype=float)
    mel_u = mel["U2"].to_numpy(dtype=float)
    if len(nv_u) < 2 or len(mel_u) < 2:
        st.error("Her gruptan en az 2 görsel gerekir.")
        st.download_button("📥 Sonuçları indir (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="ysum_sonuclar.csv", mime="text/csv")
        return

    delta = nv_u.mean() - mel_u.mean()
    t_stat, p_val = stats.ttest_ind(nv_u, mel_u, equal_var=False, alternative="greater")
    havuz = np.sqrt(((len(nv_u) - 1) * nv_u.var(ddof=1) + (len(mel_u) - 1) * mel_u.var(ddof=1))
                    / (len(nv_u) + len(mel_u) - 2))
    d = delta / havuz if havuz > 0 else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("İyi huylu − kötü huylu", f"{delta:+.3f}")
    c2.metric("Anlamlılık (p)", f"{p_val:.4f}" if np.isfinite(p_val) else "—")
    c3.metric("Fark büyüklüğü (d)", f"{d:.2f}" if np.isfinite(d) else "—")

    if not np.isfinite(p_val):
        st.warning("Veriler yeterince çeşitli değil; karşılaştırma yapılamadı.")
    elif p_val < 0.05 and delta > 0:
        st.success("İyi huylu grup, kötü huylu gruptan istatistiksel olarak daha simetrik çıktı.")
    elif p_val < 0.05 and delta <= 0:
        st.warning("Beklenenin tersine kötü huylu grup daha simetrik çıktı. Görsel seçimi, boyut veya ışık farkı etkilemiş olabilir.")
    else:
        st.info("İki grup arasında anlamlı bir fark bulunamadı.")

    goster(toplu_grafik(nv_u, mel_u))

    st.download_button("📥 Sonuçları indir (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="ysum_sonuclar.csv", mime="text/csv")
    with st.expander("Görsel bazlı sonuçlar"):
        st.dataframe(df.round(4), hide_index=True)


# ==============================================================================
# ANA SEKMELER
# ==============================================================================
tab_kavram, tab_karsilastir, tab_toplu, tab_isi, tab_sinirlar = st.tabs([
    "🔭 Kavram",
    "⚖️ Yan Yana",
    "📊 Toplu Test",
    "🌀 Isı Haritası",
    "⚠️ Sınırlılıklar",
])

# ------------------------------------------------------------------------------
# 1. KAVRAM
# ------------------------------------------------------------------------------
with tab_kavram:
    alt1, alt2, alt3 = st.tabs(["📐 Penrose", "🧮 Ölçüm", "🌐 Uygulamalar"])

    with alt1:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            kart("ok", "Penrose döşemeleri",
                 "Boşluk bırakmadan düzlemi kaplayan, ama tekrar eden bir örüntüsü olmayan "
                 "desenlerdir. 5 katlı dönel simetri içerirler.")
            kart("ok", "2011 Nobel Kimya Ödülü",
                 "Dan Shechtman, Penrose benzeri düzenli ama tekrarsız bir atom yapısını "
                 "deneysel olarak gösterdi (kuasikristal).")
        with c2:
            goster(ciz_rozet())
        with c3:
            goster(ciz_kirinim())

    with alt2:
        st.markdown(r"""
**Simetri puanı (S):** Görsel ortasındaki bir noktadan çıkan çemberler üzerinde, her noktanın
belirli bir açıyla döndürülmüş karşılığına bakılır. Karşılıklı noktalar aynı renkteyse eşleşme sayılır.
Eşleşmelerin oranı S'dir.

**Tesadüfi eşik (B):** Rastgele bir görselde de bir miktar eşleşme olur. Bu eşik, görselin açık alan
oranına göre hesaplanır: $B = p^2 + (1-p)^2$.

**Simetri puanı (U):** Tesadüfi eşik çıkarıldıktan sonra kalan düzen:

$$U = \frac{S - B}{1 - B}$$

- **U ≈ 0:** tesadüfe yakın
- **U > 0,15:** tesadüften daha simetrik
- **U < −0,15:** tesadüften daha düzensiz
""")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"Penrose oranına ($p = 1/\\varphi \\approx 0{{,}}618$) göre sabit eşik "
                f"$B \\approx {B_PENROSE:.3f}$ olur. Gerçek görseller için her görsel kendi açık alan "
                "oranından hesaplanan eşiği kullanır."
            )
        with c2:
            goster(ciz_b_egrisi())

    with alt3:
        c1, c2 = st.columns(2)
        with c1:
            kart("prop", "Dermatoloji",
                 "Lezyonun simetrisi ABCDE kuralındaki 'A' (asimetri) kriteriyle ilişkilidir. "
                 "Bu araç o ilişkiyi araştırır; tanı koymaz.")
            kart("prop", "Kültürel miras",
                 "Geleneksel İslam mimarisindeki girih desenleri de benzer simetri özellikleri taşır.")
        with c2:
            goster(ciz_abcde())

# ------------------------------------------------------------------------------
# 2. YAN YANA
# ------------------------------------------------------------------------------
with tab_karsilastir:
    st.subheader("⚖️ İki görseli karşılaştırın")

    ca, cb = st.columns(2)
    with ca:
        gri_a, ad_a = kaynak_sec("A", "1. görsel")
    with cb:
        gri_b, ad_b = kaynak_sec("B", "2. görsel")

    sa = analiz_et(gri_a, K_SEC, kaydir, B_GORSEL)
    sb = analiz_et(gri_b, K_SEC, kaydir, B_GORSEL)
    na = negatif_kontrol_u(gri_a, K_SEC, kaydir, B_GORSEL)
    nb = negatif_kontrol_u(gri_b, K_SEC, kaydir, B_GORSEL)

    ga, gb = st.columns(2)
    with ga:
        goster(isi_figuru(gri_a, sa, K_SEC, f"1. görsel · {ad_a}"))
    with gb:
        goster(isi_figuru(gri_b, sb, K_SEC, f"2. görsel · {ad_b}"))

    fark = sa["U_ort"] - sb["U_ort"]
    if not np.isfinite(fark):
        st.warning("Görsellerden biri karşılaştırılamayacak kadar düz.")
    else:
        if abs(fark) < 0.05:
            st.info("İki görsel simetri açısından birbirine çok yakın.")
        elif fark > 0:
            st.success("1. görsel, 2. görselden daha simetrik.")
        else:
            st.success("2. görsel, 1. görselden daha simetrik.")

    tablo = pd.DataFrame({
        "Ölçüm": [
            "Açık alan oranı",
            "Merkez simetri puanı",
            "Genel simetri puanı",
            "En yüksek simetri puanı",
            "Durum",
            "Kontrol (karıştırılmış görsel)",
        ],
        "1. görsel": [
            f"%{sa['p'] * 100:.1f}",
            f3(sa["U_merkez"]),
            f3(sa["U_ort"]),
            f3(sa["U_maks"]),
            duzen_yorumu(sa["U_ort"]),
            f3(na),
        ],
        "2. görsel": [
            f"%{sb['p'] * 100:.1f}",
            f3(sb["U_merkez"]),
            f3(sb["U_ort"]),
            f3(sb["U_maks"]),
            duzen_yorumu(sb["U_ort"]),
            f3(nb),
        ],
    })
    st.table(tablo)

    for etiket, n in [("1. görsel", na), ("2. görsel", nb)]:
        if np.isfinite(n) and abs(n) >= 0.15:
            st.warning(f"{etiket}: kontrol testinde de yüksek değer çıktı. Sonuç güvenilir olmayabilir.")
        elif np.isfinite(n):
            st.caption(f"{etiket}: kontrol testi temiz (sahte simetri çıkmadı).")

# ------------------------------------------------------------------------------
# 3. TOPLU TEST
# ------------------------------------------------------------------------------
with tab_toplu:
    st.subheader("📊 İyi huylu ve kötü huylu görselleri karşılaştırın")
    st.caption("Her gruptan en az 2 görsel yükleyin. Önerilen sayı grup başına 100'dür.")

    u1, u2 = st.columns(2)
    with u1:
        dos_nv = st.file_uploader("İyi huylu görseller", type=UPLOAD_TYPES,
                                  accept_multiple_files=True, key="toplu_nv")
        st.caption(f"Yüklü: {len(dos_nv)}")
    with u2:
        dos_mel = st.file_uploader("Kötü huylu görseller", type=UPLOAD_TYPES,
                                   accept_multiple_files=True, key="toplu_mel")
        st.caption(f"Yüklü: {len(dos_mel)}")

    if st.button("🚀 Testi çalıştır", type="primary"):
        if len(dos_nv) < 2 or len(dos_mel) < 2:
            st.warning("Her iki grupta da en az 2 görsel yükleyin.")
        else:
            with st.spinner("Görseller inceleniyor..."):
                st.session_state["toplu_kayit"] = toplu_calistir(dos_nv, dos_mel, kaydir, B_GORSEL)

    if "toplu_kayit" in st.session_state:
        toplu_sonuc_goster(st.session_state["toplu_kayit"])

# ------------------------------------------------------------------------------
# 4. ISI HARİTASI
# ------------------------------------------------------------------------------
with tab_isi:
    st.subheader("🌀 Simetri haritası")
    st.caption("Kırmızı: daha simetrik bölge · Mavi: daha simetrik olmayan bölge")

    c_sol, _ = st.columns([1, 2])
    with c_sol:
        k_isi = st.selectbox("Simetri türü", [2, 5, 6, 8], index=1,
                             format_func=lambda v: f"{v} kat")

    gri_i, ad_i = kaynak_sec("isi", "Görsel")
    si = analiz_et(gri_i, k_isi, kaydir, B_GORSEL)

    m1, m2, m3 = st.columns(3)
    m1.metric("Açık alan oranı", f"%{si['p'] * 100:.1f}")
    m2.metric("Merkez simetri puanı", f3(si["U_merkez"]))
    m3.metric("Genel simetri puanı", f3(si["U_ort"]))
    st.caption(f"Durum: **{duzen_yorumu(si['U_ort'])}**")

    g1, g2 = st.columns(2)
    with g1:
        st.image(gri_i, caption=ad_i, width=280, clamp=True)
    with g2:
        goster(isi_figuru(gri_i, si, k_isi, ad_i))

    U = si["U_izgara"]
    if np.isfinite(U).any():
        duz = np.where(np.isfinite(U), U, -np.inf).ravel()
        idx = np.argsort(duz)[::-1][:3]
        satirlar = []
        for sira, fi in enumerate(idx, 1):
            iy, ix = np.unravel_index(fi, U.shape)
            satirlar.append(f"- **{sira}.** x = {si['xs'][ix]}, y = {si['ys'][iy]} (puan {U[iy, ix]:.2f})")
        st.markdown("**En simetrik üç nokta**")
        st.markdown("\n".join(satirlar))

# ------------------------------------------------------------------------------
# 5. SINIRLILIKLAR
# ------------------------------------------------------------------------------
with tab_sinirlar:
    st.subheader("⚠️ Sınırlılıklar")
    st.markdown("""
- Bu uygulama **tıbbi tanı aracı değildir**. Sonuçlar klinik karar yerine geçmez.
- Lezyonun sınırı otomatik çıkarılmaz; ölçüm sabit bir merkez etrafında yapılır.
- Görseller 200 × 200 piksele küçültülür ve gri tona çevrilir.
- Örneklem küçüktür; sonuçlar genellenemez.
- HAM10000 veri seti yalnızca ticari olmayan kullanıma açıktır.
""")

st.markdown("---")
st.caption("⚠️ Araştırma ve eğitim amaçlıdır. Tıbbi tanı, tarama veya karar için kullanılamaz.")
