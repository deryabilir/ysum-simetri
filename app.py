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
P_PENROSE = 1 / PHI                                   # p = 1/φ ≈ 0.618 (model varsayımı)
B_PENROSE = P_PENROSE ** 2 + (1 - P_PENROSE) ** 2     # ≈ 0.528

K_SECENEKLER = {
    "k=2 · 180° (dermatoskopi protokolü)": 2,
    "k=5 · 72° (Penrose)": 5,
    "k=6 · 60°": 6,
    "k=8 · 45°": 8,
}
IMG_N = 200                     # tüm görseller bu boyuta getirilir
RADII = (20, 40, 60, 80)        # merkez skor yarıçapları (piksel)
N_ACI = 36                      # açı örnekleme sayısı (10° adım)
HEAT_RADII = (10, 20, 30, 40)   # ısı haritası yarıçapları
HEAT_ADIM = 8                   # ısı haritası ızgara adımı (piksel)
UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp"]

ORNEK_SIMETRIK = "Sentetik · Yuvarlak şekil (simetrik)"
ORNEK_ASIMETRIK = "Sentetik · Düzensiz şekil (asimetrik)"
ORNEK_5KAT = "Sentetik · 5 katlı desen"
ORNEK_GURULTU = "Sentetik · Rastgele gürültü"
ORNEKLER = [ORNEK_SIMETRIK, ORNEK_ASIMETRIK, ORNEK_5KAT, ORNEK_GURULTU]

# ==============================================================================
# STİL
# ==============================================================================
st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #0b1020 0%, #1e1b4b 55%, #0e7490 100%);
       padding: 28px 32px; border-radius: 18px; color: #f8fafc;
       box-shadow: 0 8px 30px rgba(0,0,0,.25); margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2.1rem;}
.hero p {margin: 0; opacity: .92; line-height: 1.5;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 999px; margin: 10px 8px 0 0;
        font-size: .82rem; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);}
.kart {border-radius: 14px; padding: 14px 18px; margin: 10px 0; border-left: 5px solid;
       background: rgba(127,127,127,.07);}
.kart .etiket {font-size: .70rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 700; opacity: .85;}
.kart h4 {margin: 4px 0 6px 0;}
.kart-ok {border-color: #22d3ee;}
.kart-prop {border-color: #fbbf24;}
.kart-res {border-color: #f43f8e;}
.kart-warn {border-color: #ef4444;}
</style>
""", unsafe_allow_html=True)

ETIKETLER = {
    "ok": ("Kaynaklı bilgi", "kart-ok"),
    "prop": ("Öneri · doğrulanmadı", "kart-prop"),
    "res": ("Araştırma sorusu", "kart-res"),
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
  <p>Penrose döşemelerinden türetilen şans-arındırılmış simetri ölçümü (U skoru) ve
  dermatoskopik lezyonlarda ABCDE "A" (asimetri) kriteriyle ilişkisinin araştırılması.</p>
  <span class="badge">Penrose referans B = 0.528</span>
  <span class="badge">Katlı simetri: k = 2 · 5 · 6 · 8</span>
  <span class="badge">Şans-arındırılmış skor: U</span>
  <span class="badge">Araştırma amaçlıdır · tıbbi tanı aracı değildir</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# YAN MENÜ
# ==============================================================================
st.sidebar.header("⚙️ Analiz Ayarları")

k_etiket = st.sidebar.selectbox(
    "Simetri kat sayısı (k) · Yan Yana Analiz için",
    list(K_SECENEKLER.keys()),
    index=0,
)
K_SEC = K_SECENEKLER[k_etiket]

b_modu = st.sidebar.radio(
    "Şans eşiği (B) kaynağı",
    ["Görsel bazlı B = p² + (1−p)² (önerilen)", "Penrose referans B = 0.528"],
    index=0,
)
B_GORSEL = b_modu.startswith("Görsel")

kaydir = st.sidebar.checkbox(
    "Merkezi koyu bölgenin ağırlık merkezine kaydır",
    value=False,
    help="Kapalıyken merkez görüntünün geometrik merkezidir (100, 100). "
         "Açıksa koyu piksellerin ağırlık merkezi kullanılır. Bu bir segmentasyon değildir.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Kullanılan formüller**")
st.sidebar.latex(r"S_k=\frac{\text{eşleşen çift}}{\text{geçerli çift}}")
st.sidebar.latex(r"U_k=\frac{S_k-B}{1-B}")
st.sidebar.caption("U ≈ 0: şans düzeyi · U > 0: şans üstü düzen · U < 0: şans altı")
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Toplu Test sekmesi her zaman k = 2 (dermatoskopi protokolü) kullanır. "
    "Bu kenar çubuğundaki k yalnızca Yan Yana Analiz içindir."
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
    """Bozuk veya desteklenmeyen dosyada uygulamayı çökertmez; (görsel, hata) döndürür."""
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
        # Her terim 72° dönmede değişmez: cos(5θ), cos(10θ) ve r'ye bağlı terim.
        # Bu desen yalnızca merkez (100, 100) etrafında 5 katlıdır.
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
    """Görsel bazlı B = p²+(1−p)²; p∈{0,1} ise hesaplanamaz (None)."""
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
    """Vektörize simetri oranı. Her merkez için S_k döndürür (nokta başına)."""
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
    """Tek görsel için merkez skoru ve (istenirse) ısı haritası ızgara özetini hesaplar."""
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
    """Pikselleri rastgele karıştırır (tohum 0). Merkez ORİJİNAL görselden alınır."""
    ikili_orijinal, _ = ikili_matris(gri)
    merkez = merkez_sec(ikili_orijinal, kaydir_on)
    rng = np.random.default_rng(0)
    karisik = rng.permutation(gri.ravel()).reshape(gri.shape)
    return analiz_et(karisik, k, kaydir_on, b_gorsel, merkez=merkez)["U_ort"]


def duzen_yorumu(U):
    if not np.isfinite(U):
        return "Hesaplanamadı (görsel düz olabilir)"
    if U >= 0.15:
        return "Şans üstü düzen"
    if U <= -0.15:
        return "Şans altı (belirgin düzensizlik)"
    return "Şans düzeyine yakın"


# ==============================================================================
# ÇİZİMLER (illüstrasyon; veri içermez)
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
    ax.set_title("5 katlı rozet (illüstrasyon)", color="white", fontsize=9)
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
    ax.set_title("Kırınım deseni (illüstrasyon)", color="white", fontsize=9)
    return fig


def ciz_girih():
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=110)
    aci = np.arange(10) * 2 * np.pi / 10 + np.pi / 2
    nokta = np.c_[np.cos(aci), np.sin(aci)]
    yol = [(3 * j) % 10 for j in range(10)]          # {10/3} yıldız yolu
    ax.add_patch(Polygon(nokta[yol], closed=True, facecolor="#14b8a6", alpha=0.30,
                         edgecolor="#0f766e", lw=2))
    ax.add_patch(Polygon(nokta, closed=True, fill=False, edgecolor="#0f766e", lw=1, ls="--"))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("10 katlı {10/3} yıldız (illüstrasyon)", fontsize=9)
    return fig


def ciz_b_egrisi():
    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=110)
    p = np.linspace(0, 1, 200)
    ax.plot(p, p ** 2 + (1 - p) ** 2, color="#2563eb", lw=2,
            label=r"$B(p)=p^2+(1-p)^2$")
    ax.axvline(P_PENROSE, color="#fbbf24", ls="--", label=r"$p=1/\varphi\approx 0{,}618$")
    ax.axhline(B_PENROSE, color="#f43f8e", ls="--", label=r"$B\approx 0{,}528$")
    ax.scatter([P_PENROSE], [B_PENROSE], color="#f43f8e", zorder=5)
    ax.set_xlabel("Açık piksel oranı (p)", fontsize=8)
    ax.set_ylabel("Şans eşiği (B)", fontsize=8)
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
        (axs[0], r_sim, "#16a34a", "Simetrik (sentetik)"),
        (axs[1], r_asim, "#dc2626", "Asimetrik (sentetik)"),
    ]:
        ax.fill(r * np.cos(th), r * np.sin(th), color=renk, alpha=0.25)
        ax.plot(r * np.cos(th), r * np.sin(th), color=renk, lw=1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(baslik, fontsize=8)
    fig.suptitle("ABCDE 'A' — sentetik şekiller (gerçek lezyon değildir)", fontsize=8)
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
    cb.set_label("U (şans-arındırılmış)", fontsize=8)
    ax.set_title(f"{baslik}\nk={k} · Ort U = {f3(sonuc['U_ort'])}", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ==============================================================================
# GİRDİ SEÇİCİ
# ==============================================================================
def kaynak_sec(anahtar, baslik):
    dosya = st.file_uploader(
        f"{baslik} — görsel yükleyin", type=UPLOAD_TYPES, key=f"up_{anahtar}"
    )
    ornek = st.selectbox("Veya örnek seçin", ORNEKLER, key=f"ornek_{anahtar}")
    if dosya is not None:
        gri, hata = gri_oku_guvenli(dosya)
        if gri is None:
            st.error(f"{dosya.name} {hata}. Bunun yerine seçili örnek kullanılıyor.")
            return ornek_desen(ornek), ornek
        st.caption("Yüklenen görsel kullanılıyor; örnek seçimi yok sayılır.")
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
        "n": int(len(u)),
        "Ort. S₂": s.mean() if len(s) else np.nan,
        "Std. S₂": s.std(ddof=1) if len(s) > 1 else np.nan,
        "Min S₂": s.min() if len(s) else np.nan,
        "Maks S₂": s.max() if len(s) else np.nan,
        "Ort. U₂": u.mean() if len(u) else np.nan,
        "Std. U₂": u.std(ddof=1) if len(u) > 1 else np.nan,
    }


def toplu_grafik(nv_u, mel_u):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
    axs[0].boxplot([nv_u, mel_u], showmeans=True)
    axs[0].set_xticks([1, 2])
    axs[0].set_xticklabels(["İyi huylu (nv)", "Kötü huylu (mel)"])
    axs[0].axhline(0, color="gray", ls=":", lw=1)
    axs[0].set_ylabel("U₂")
    axs[0].set_title("Dağılım (kutu grafiği)", fontsize=9)
    axs[1].hist(nv_u, bins=15, alpha=0.6, label="nv", color="#22d3ee")
    axs[1].hist(mel_u, bins=15, alpha=0.6, label="mel", color="#f43f8e")
    axs[1].axvline(0, color="gray", ls=":", lw=1)
    axs[1].set_xlabel("U₂")
    axs[1].set_title("Histogram", fontsize=9)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def toplu_sonuc_goster(kayit):
    df = kayit["df"]
    st.caption(
        f"Son çalıştırma ayarları: k=2 · merkez "
        f"{'koyu bölgenin ağırlık merkezi' if kayit['kaydir'] else '(100, 100)'} · "
        f"B {'görsel bazlı' if kayit['b_gorsel'] else 'Penrose referans (0,528)'}. "
        "Ayarlar veya dosyalar değiştiyse yeniden çalıştırın."
    )

    hatali = df[df["Hata"] != ""]
    if len(hatali):
        st.warning(f"{len(hatali)} dosya işlenemedi; istatistiğe dahil edilmedi.")
        with st.expander("İşlenemeyen dosyalar"):
            st.dataframe(hatali[["Dosya", "Grup", "Hata"]], hide_index=True)

    gecerli = df.dropna(subset=["U2"])
    nv = gecerli[gecerli["Grup"] == "nv"]
    mel = gecerli[gecerli["Grup"] == "mel"]

    st.markdown("### 📋 Grup özeti")
    ozet = pd.DataFrame([
        {"Grup": "İyi huylu (nv)", **grup_ozeti(nv)},
        {"Grup": "Kötü huylu (mel)", **grup_ozeti(mel)},
    ])
    st.dataframe(ozet.round(3), hide_index=True)

    st.markdown("### 🧪 İstatistiksel karşılaştırma")
    st.caption(
        "H0: μ(U₂, nv) = μ(U₂, mel) · H1: μ(U₂, nv) > μ(U₂, mel) · Tek yönlü Welch t-testi. "
        "Test U₂ üzerinde yapılır; çünkü B her görsel için ayrı hesaplanır ve ham S₂ "
        "değerleri görselin p değerine bağlıdır."
    )

    nv_u = nv["U2"].to_numpy(dtype=float)
    mel_u = mel["U2"].to_numpy(dtype=float)
    if len(nv_u) < 2 or len(mel_u) < 2:
        st.error("Her grupta en az 2 geçerli görsel gerekir (önerilen: 100).")
        st.download_button(
            "📥 Görsel bazlı sonuçları indir (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ysum_gorsel_bazli_sonuclar.csv",
            mime="text/csv",
        )
        return

    delta = nv_u.mean() - mel_u.mean()
    t_stat, p_val = stats.ttest_ind(nv_u, mel_u, equal_var=False, alternative="greater")
    havuz = np.sqrt(((len(nv_u) - 1) * nv_u.var(ddof=1) + (len(mel_u) - 1) * mel_u.var(ddof=1))
                    / (len(nv_u) + len(mel_u) - 2))
    d = delta / havuz if havuz > 0 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ΔU (nv − mel)", f"{delta:+.3f}")
    c2.metric("t", f"{t_stat:.3f}" if np.isfinite(t_stat) else "—")
    c3.metric("p (tek yönlü)", f"{p_val:.4f}" if np.isfinite(p_val) else "—")
    c4.metric("Cohen's d", f"{d:.3f}" if np.isfinite(d) else "—")

    if not np.isfinite(p_val):
        st.warning("Varyans sıfır olduğu için test hesaplanamadı.")
    elif p_val < 0.05 and delta > 0:
        st.success(
            "H1 yönünde istatistiksel olarak anlamlı fark bulundu (p < 0,05). "
            "Ancak bu tek bir merkez, sınırlı bir örneklem ve tek bir protokolle elde edildi. "
            "Klinik geçerlilik anlamına gelmez; tanı için kullanılamaz."
        )
    elif p_val < 0.05 and delta <= 0:
        st.warning(
            "Anlamlı fark var ama H1 ile ters yönde: kötü huylu grubun U₂ değeri daha yüksek. "
            "Bu beklentiyle çelişir; boyut, aydınlatma, kaynak farkı ve seçim yanlılığı gibi "
            "açıklamalar araştırılmalıdır."
        )
    else:
        st.info(
            "p ≥ 0,05: H0 reddedilemedi. Bu, iki grubun eşit olduğu anlamına gelmez; "
            "örneklem gücü yetersiz olabilir."
        )

    goster(toplu_grafik(nv_u, mel_u))

    st.download_button(
        "📥 Görsel bazlı sonuçları indir (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ysum_gorsel_bazli_sonuclar.csv",
        mime="text/csv",
    )
    with st.expander("Görsel bazlı ham sonuçlar"):
        st.dataframe(df.round(4), hide_index=True)


# ==============================================================================
# ANA SEKMELER
# ==============================================================================
tab_kavram, tab_karsilastir, tab_toplu, tab_isi, tab_kaynak = st.tabs([
    "🔭 Kavram & Tarih",
    "⚖️ Yan Yana Analiz",
    "📊 Toplu Test",
    "🌀 Isı Haritası",
    "📚 Kaynaklar & Sınırlılıklar",
])

# ------------------------------------------------------------------------------
# 1. KAVRAM & TARİH
# ------------------------------------------------------------------------------
with tab_kavram:
    alt1, alt2, alt3, alt4 = st.tabs([
        "📐 Penrose & Kuasikristaller",
        "🧮 Matematik (YSÜM)",
        "🕌 Girih & Tarih",
        "🌐 Uygulama Alanları",
    ])

    with alt1:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            kart("ok", "Aperiyodik döşemeler",
                 "Roger Penrose (1974), düzlemi boşluk ve çakışma olmadan kaplayan ama öteleme "
                 "simetrisi taşımayan iki rombus türü tanımladı. Bu döşemelerde 5 katlı dönel "
                 "simetri görülür.")
            kart("ok", "Kuasikristaller ve 2011 Nobel Kimya Ödülü",
                 "Shechtman ve arkadaşları (1984), periyodik olmayan ama düzenli yönelim düzenine "
                 "sahip bir alaşım rapor etti. Bu deneysel keşif 2011 Nobel Kimya Ödülü ile "
                 "ödüllendirildi (Dan Shechtman). Penrose tipi döşemeler, bu yapıların "
                 "modellenmesinde kullanılan matematiksel modellerden biridir.")
            kart("warn", "Kaynak ayrımı",
                 "Penrose'un 1974 çalışması matematiksel bir katkıdır. Nobel ödülü ise kuasikristal "
                 "deneysel keşfine verilmiştir. Bu ikisi aynı şey değildir.")
        with c2:
            goster(ciz_rozet())
        with c3:
            goster(ciz_kirinim())
        st.caption("Rozet ve kırınım deseni illüstrasyondur; veri içermez.")

    with alt2:
        st.markdown(r"""
### Simetri sayısı ve şans arındırması

Görüntü, ortalama parlaklığa göre ikili (0/1) matrise dönüştürülür. Merkez noktanın etrafında
yarıçapları $r \in \{20, 40, 60, 80\}$ piksel olan çemberler üzerinde 36 nokta seçilir. Her noktanın
$2\pi/k$ kadar döndürülmüş karşılığı alınır. Eşleşen çiftlerin oranı simetri sayısıdır:

$$S_k=\frac{\#\{\text{eşleşen çift}\}}{\#\{\text{geçerli çift}\}}$$

Açık piksel oranı $p$ olan ve birbirinden bağımsız varsayılan piksellerde, rastgele bir çiftin
eşleşme olasılığı şudur:

$$B(p)=p^2+(1-p)^2$$

Şans-arındırılmış simetri skoru:

$$U_k=\frac{S_k-B}{1-B}$$

- $U_k \approx 0$: şans düzeyi
- $U_k > 0$: şans üstü düzen
- $U_k < 0$: şans altı (belirgin düzensizlik)
""")

        st.markdown(r"""
### Penrose tarafında $B$ nereden geliyor?

1. Standart Penrose rombus büyüme kuralında kalın ($K_n$) ve ince ($I_n$) karo sayıları her
   nesilde şöyle değişir ($T_n = K_n + I_n$):
$$K_{n+1}=2K_n+I_n,\qquad I_{n+1}=K_n+I_n$$
2. Bu doğrusal dönüşümün matrisi $M=\begin{pmatrix}2&1\\1&1\end{pmatrix}$ ve karakteristik
   denklemi $\lambda^2-3\lambda+1=0$'dır. En büyük özdeğer $\lambda_1=\varphi^2\approx 2{,}618$'dir.
3. Oranlar $K_n/I_n \to \varphi$ yakınsar. Buradan $K_n/T_n \to \varphi/(1+\varphi)=1/\varphi\approx 0{,}618$.
4. **Model varsayımı:** $p=1/\varphi$ alınır. Bu, döşeme karo oranını görüntüdeki açık piksel
   oranına bağlayan bir modelleme tercihidir. Kanıtlanmış bir teorem değildir.
5. Bu değerle $B = \dfrac{1}{\varphi^2}+\dfrac{1}{\varphi^4}\approx 0{,}528$.
""")
        kart("prop", "Model sınırı",
             "Gerçek bir dermatoskopik görüntüde açık piksel oranının 1/φ olması beklenmez. "
             "Bu yüzden önerilen modda B her görsel için kendi p değerinden hesaplanır. "
             "Penrose referans değeri (0,528) yalnızca karşılaştırma için sunulur.")

        fib_satirlar = []
        K, I = 1, 0
        for n in range(1, 7):
            K, I = 2 * K + I, K + I
            T = K + I
            fib_satirlar.append({
                "n": n,
                "K_n": K,
                "I_n": I,
                "T_n": T,
                "K_n / T_n": f"{K / T:.6f}",
            })
        st.markdown("#### Döşeme sayıları ve oran yakınsaması")
        st.table(pd.DataFrame(fib_satirlar))
        st.caption(f"Limit değer: 1/φ = {1 / PHI:.6f}. K_n/T_n dizisi bu değere yakınsar.")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Şans eşiği eğrisi")
            st.markdown(
                "$B(p)$ eğrisi $p=0{,}5$ noktasında en küçük değeri alır ($B=0{,}5$). "
                "Penrose referans noktası ($p=1/\\varphi$) bu eğri üzerinde $B\\approx 0{,}528$ "
                "değerindedir."
            )
        with c2:
            goster(ciz_b_egrisi())

    with alt3:
        c1, c2 = st.columns([2, 1])
        with c1:
            kart("ok", "Girih çinileri",
                 "Lu ve Steinhardt (2007), ortaçağ İslam mimarisindeki girih desenlerinin on katlı "
                 "simetri ve Penrose benzeri kuasikristal düzenler içerdiğini gösterdi.")
            kart("warn", "Bu projenin katkısı değildir",
                 "Girih–Penrose benzerliği bu projenin keşfi değildir. Proje bu ilişkiyi "
                 "kaynaktan aktarır; yeni bir tarihsel iddia öne sürmez.")
            kart("prop", "Restorasyon kullanımı",
                 "Yıpranmış yüzeylerde simetri merkezlerinin bu yöntemle bulunabileceği bir "
                 "öneridir. Bu aşamada böyle bir uygulama test edilmemiştir.")
        with c2:
            goster(ciz_girih())
            st.caption("{10/3} yıldızı illüstrasyondur.")

    with alt4:
        c1, c2, c3 = st.columns(3)
        with c1:
            kart("ok", "Malzeme bilimi",
                 "Kuasikristal alaşımların yapısı ve yüzey özellikleri malzeme bilimi "
                 "literatürünün aktif konularıdır.")
            kart("prop", "Ticari ürün iddiaları",
                 "Kuasikristal kaplamalar üzerine çalışmalar vardır. Bu projede ürün performansı "
                 "iddiası doğrulanmamıştır.")
        with c2:
            kart("prop", "Dermatolojik asimetri",
                 "Nachbar ve ark. (1994) ABCD kuralını tanımladı. A (asimetri) kriteri bu "
                 "projenin ölçtüğü kavramla ilgilidir. Yöntemin tanıdaki başarısı bu projede "
                 "test edilmemiştir.")
            goster(ciz_abcde())
        with c3:
            kart("prop", "Desen analizi ve kültürel miras",
                 "Simetri ölçümünün tarihî desen arşivlerinde yardımcı olabileceği bir öneridir. "
                 "Bu doğrulanmış bir uygulama değildir.")

# ------------------------------------------------------------------------------
# 2. YAN YANA ANALİZ
# ------------------------------------------------------------------------------
with tab_karsilastir:
    st.subheader("⚖️ İki Görselin Yan Yana Simetri Karşılaştırması")
    st.caption(
        "Her görsel için merkez skoru (U_merkez) ve ızgara ortalaması (U_ort) hesaplanır. "
        "Negatif kontrol, pikselleri rastgele karıştırarak yapısal bilgiyi yok eder; "
        "bu durumda U ≈ 0 beklenir."
    )
    st.info(
        "Kontrol önerisi: 'Sentetik · 5 katlı desen' seçip k=5 ile çalıştırın. Bu desen yalnızca "
        "merkezinde 5 katlıdır, bu nedenle asıl beklenti U_merkez'in yüksek çıkmasıdır. "
        "Izgara ortalaması (U_ort) bu desende belirgin yükselmeyebilir; bu beklenen bir "
        "davranıştır, hata değildir."
    )

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

    tablo = pd.DataFrame({
        "Metrik": [
            "Açık piksel oranı (p)",
            "Şans eşiği (B)",
            "Merkez S_k",
            "U_merkez",
            "U_ort (ızgara)",
            "U_maks (ızgara)",
            "Yorum (U_ort)",
            "Negatif kontrol U_ort",
        ],
        "1. görsel": [
            f"%{sa['p'] * 100:.1f}",
            f3(sa["B"]) if sa["B"] is not None else "—",
            f3(sa["S_merkez"]),
            f3(sa["U_merkez"]),
            f3(sa["U_ort"]),
            f3(sa["U_maks"]),
            duzen_yorumu(sa["U_ort"]),
            f3(na),
        ],
        "2. görsel": [
            f"%{sb['p'] * 100:.1f}",
            f3(sb["B"]) if sb["B"] is not None else "—",
            f3(sb["S_merkez"]),
            f3(sb["U_merkez"]),
            f3(sb["U_ort"]),
            f3(sb["U_maks"]),
            duzen_yorumu(sb["U_ort"]),
            f3(nb),
        ],
    })
    st.table(tablo)

    st.markdown("### 🔎 Karşılaştırma")
    fark_m = sa["U_merkez"] - sb["U_merkez"]
    fark_o = sa["U_ort"] - sb["U_ort"]
    if not np.isfinite(fark_o):
        st.warning("Bir veya iki görsel için fark hesaplanamadı (düz görsel olabilir).")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Fark · U_merkez (1 − 2)", f3(fark_m))
        m2.metric("Fark · U_ort (1 − 2)", f3(fark_o))
        if abs(fark_o) < 0.05:
            st.info("Izgara ortalamaları arasındaki fark 0,05 pratik eşiğinin altında.")
        elif fark_o > 0:
            st.info("1. görselin ızgara ortalaması daha yüksek; yani daha fazla şans üstü düzen gösteriyor.")
        else:
            st.info("2. görselin ızgara ortalaması daha yüksek; yani daha fazla şans üstü düzen gösteriyor.")

    for etiket, n in [("1. görsel", na), ("2. görsel", nb)]:
        if np.isfinite(n) and abs(n) >= 0.15:
            st.warning(
                f"{etiket} için negatif kontrol U = {n:.3f}. Karıştırılmış görselde de belirgin "
                "bir değer çıktı. Bu, görsel dokusunun yöntemi yanlı etkilediğini düşündürür; "
                "sonuç dikkatle okunmalıdır."
            )
        elif np.isfinite(n):
            st.success(f"{etiket} için negatif kontrol beklendiği gibi U ≈ 0 (U = {n:.3f}).")

    st.caption(
        "Bu karşılaştırma tek bir görsel çiftine dayanır. 0,05 eşiği pratik bir eşiktir, "
        "istatistiksel anlam taşımaz. Sonuçlar tanı, tarama veya karar için kullanılamaz."
    )

# ------------------------------------------------------------------------------
# 3. TOPLU TEST
# ------------------------------------------------------------------------------
with tab_toplu:
    st.subheader("📊 Toplu Test · İyi huylu (nv) ve kötü huylu (mel) grupları")
    st.info(
        "Her grupta en az 2 görsel gerekir; önerilen sayı grup başına 100'dür. Görseller "
        "kendi bilgisayarınızdan yüklenir (ISIC Archive veya HAM10000 gibi kaynaklardan). "
        "Sonuçlar bu gerçek görsellere dayanır; sabit veya örnek değerler kullanılmaz."
    )

    u1, u2 = st.columns(2)
    with u1:
        dos_nv = st.file_uploader(
            "İyi huylu (nv) görseller", type=UPLOAD_TYPES,
            accept_multiple_files=True, key="toplu_nv",
        )
        st.caption(f"Yüklü: {len(dos_nv)} görsel")
    with u2:
        dos_mel = st.file_uploader(
            "Kötü huylu (mel) görseller", type=UPLOAD_TYPES,
            accept_multiple_files=True, key="toplu_mel",
        )
        st.caption(f"Yüklü: {len(dos_mel)} görsel")

    if st.button("🚀 Toplu Testi Çalıştır", type="primary"):
        if len(dos_nv) < 2 or len(dos_mel) < 2:
            st.warning("Her iki grupta da en az 2 görsel yükleyin.")
        else:
            with st.spinner("Görseller analiz ediliyor..."):
                st.session_state["toplu_kayit"] = toplu_calistir(dos_nv, dos_mel, kaydir, B_GORSEL)

    if "toplu_kayit" in st.session_state:
        toplu_sonuc_goster(st.session_state["toplu_kayit"])

# ------------------------------------------------------------------------------
# 4. ISI HARİTASI
# ------------------------------------------------------------------------------
with tab_isi:
    st.subheader("🌀 Yerel Simetri Isı Haritası")
    st.caption(
        "Izgaranın her noktasında küçük yarıçaplı (10–40 piksel) çevresel simetri ölçülür. "
        "Kırmızı tonlar yüksek U (şans üstü düzen), mavi tonlar düşük U (şans altı) anlamına gelir."
    )

    c_sol, _ = st.columns([1, 2])
    with c_sol:
        k_isi = st.selectbox(
            "Kat sayısı (k)", [2, 5, 6, 8], index=1,
            format_func=lambda v: f"k={v}",
        )

    gri_i, ad_i = kaynak_sec("isi", "Görsel")
    si = analiz_et(gri_i, k_isi, kaydir, B_GORSEL)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Açık piksel (p)", f"%{si['p'] * 100:.1f}")
    m2.metric("Şans eşiği (B)", f3(si["B"]) if si["B"] is not None else "—")
    m3.metric("U_merkez", f3(si["U_merkez"]))
    m4.metric("U_ort", f3(si["U_ort"]))
    m5.metric("U_maks", f3(si["U_maks"]))

    g1, g2 = st.columns([1, 1])
    with g1:
        st.image(gri_i, caption=f"Görsel · {ad_i}", width=280, clamp=True)
    with g2:
        goster(isi_figuru(gri_i, si, k_isi, ad_i))

    U = si["U_izgara"]
    if np.isfinite(U).any():
        duz = np.where(np.isfinite(U), U, -np.inf).ravel()
        idx = np.argsort(duz)[::-1][:3]
        satirlar = []
        for sira, fi in enumerate(idx, 1):
            iy, ix = np.unravel_index(fi, U.shape)
            satirlar.append(
                f"- **{sira}. odak:** (x = {si['xs'][ix]}, y = {si['ys'][iy]}) · U = {U[iy, ix]:.3f}"
            )
        st.markdown("#### En yüksek U'lu noktalar")
        st.markdown("\n".join(satirlar))
    else:
        st.warning("Bu görsel için ızgara hesabı yapılamadı (görsel düz olabilir).")

# ------------------------------------------------------------------------------
# 5. KAYNAKLAR & SINIRLILIKLAR
# ------------------------------------------------------------------------------
with tab_kaynak:
    st.subheader("📚 Kaynakça")
    st.markdown("""
- Penrose, R. (1974). The role of aesthetics in pure and applied mathematical research. *Bulletin of the Institute of Mathematics and its Applications*, 10, 266–271.
- Shechtman, D., Blech, I., Gratias, D., & Cahn, J. W. (1984). Metallic phase with long-range orientational order and no translational symmetry. *Physical Review Letters*, 53(20), 1951–1953.
- Royal Swedish Academy of Sciences (2011). *The Nobel Prize in Chemistry 2011* (Dan Shechtman).
- Lu, P. J., & Steinhardt, P. J. (2007). Decagonal and quasi-crystalline tilings in medieval Islamic architecture. *Science*, 315(5815), 1106–1110.
- Nachbar, F., ve ark. (1994). The ABCD rule of dermatoscopy: High prospective value in the diagnosis of doubtful melanocytic skin lesions. *Journal of the American Academy of Dermatology*, 30(4), 551–559.
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161. (Lisans: CC BY-NC 4.0)
- ISIC Archive. https://www.isic-archive.com
""")

    st.subheader("⚠️ Sınırlılıklar")
    kart("warn", "Segmentasyon yok",
         "Lezyon sınırı çıkarılmıyor. Ölçüm sabit bir merkez etrafında yapılıyor.")
    kart("warn", "Merkez varsayılanı",
         "Varsayılan merkez görüntünün geometrik merkezidir. Koyu bölgenin ağırlık merkezi kaba bir "
         "vekildir ve segmentasyon yerine geçmez.")
    kart("warn", "İkili eşik",
         "Gri tonlu görüntü ortalama parlaklığa göre ikiliye çevrilir. Renk ve ışık farkları "
         "dikkate alınmaz.")
    kart("warn", "Bağımsızlık varsayımı",
         "B = p² + (1−p)² komşu pikseller bağımsızmış gibi hesaplanır. Komşu pikseller ise "
         "korelidir; B bu nedenle bir yaklaşımdır.")
    kart("warn", "Örneklem ve seçim yanlılığı",
         "Toplu test klinik araştırma standardının çok altındadır. Görsel seçim yöntemi sonucu "
         "etkileyebilir.")
    kart("warn", "Penrose referansı",
         "B ≈ 0,528 bir model değeridir (p = 1/φ varsayımı). Görsel verisiyle doğrulanmamıştır.")
    kart("warn", "Girih–Penrose ilişkisi",
         "Bu ilişki Lu ve Steinhardt (2007) kaynağından aktarılmıştır; bu projenin keşfi değildir.")
    kart("warn", "Tıbbi kullanım",
         "Bu uygulama tanı, tarama veya karar aracı değildir. Klinik geçerliliği yoktur.")
    kart("warn", "Veri lisansı",
         "HAM10000 CC BY-NC 4.0 lisansıyla yalnızca ticari olmayan kullanıma açıktır. ISIC "
         "görselleri için kendi lisanslarını kontrol edin.")
    kart("warn", "Çözünürlük",
         "Görseller 200 × 200 piksele yeniden boyutlandırılır; en-boy oranı korunmaz.")

    st.subheader("🚫 Bu sürümde yer almayan iddialar")
    st.markdown("""
- Önceden yazılmış sabit istatistik değerleri (ör. "SNR", "p < 0,001", "%99,9") yoktur. Tüm sayılar yüklenen görsellerden hesaplanır.
- "0,1 saniyede tespit" gibi hız iddiaları yoktur.
- Fabrika PASS/FAIL kararı yoktur.
- "Literatürde ilk kez" ve "dermatolojide ilk" gibi özgünlük iddiaları yoktur.
- Restorasyon koordinat rehberi ve belirli bir yapıya ait tarihsel örnek yoktur.
""")

st.markdown("---")
st.caption(
    "⚠️ Bu uygulama bir tıbbi teşhis aracı değildir. Araştırma ve eğitim amaçlıdır; "
    "sonuçlar klinik karar yerine geçmez."
)
