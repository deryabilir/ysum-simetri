import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import streamlit as st
from scipy import stats

# =============================================================================
# SABİTLER
# =============================================================================
PHI = (1 + 5 ** 0.5) / 2
P_PENROSE = 1 / PHI
B_PENROSE = P_PENROSE ** 2 + (1 - P_PENROSE) ** 2          # ≈ 0,528

IMG_N = 200                          # tüm görseller 200×200 gri tonlamaya dönüştürülür
IZGARA_BASI = 10
IZGARA_ADIMI = 8
ACI_SAYISI = 36
SKOR_YARICAPLARI = (20, 40, 60, 80)  # S_k ve U_k hesabı
ISI_YARICAPLARI = (10, 20, 30, 40)   # ısı haritası
MERKEZ = (IMG_N // 2, IMG_N // 2)
YORUM_ESIGI = 0.15                   # okuma kolaylığı için proje konvansiyonu; istatistiksel eşik değildir
ALFA = 0.05
GRUP_IYI = "İyi huylu (nv)"
GRUP_KOTU = "Kötü huylu (mel)"
UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp"]
K_SECENEKLER = {
    2: "2 kat · 180° (dermatoskopi)",
    5: "5 kat · 72° (Penrose)",
    6: "6 kat · 60°",
    8: "8 kat · 45°",
}
ORNEKLER = {
    "Yuvarlak şekil": "yuvarlak",
    "Düzensiz şekil": "duzensiz",
    "5 katlı desen": "bes_kat",
    "Rastgele gürültü": "gurultu",
}

# =============================================================================
# SAYFA VE KENAR ÇUBUĞU
# =============================================================================
st.set_page_config(page_title="YSÜM · Yerel Simetri Üstünlük Modeli", page_icon="🌀", layout="wide")
st.title("🌀 YSÜM · Yerel Simetri Üstünlük Modeli")
st.caption("Simetri ölçümünü şans beklentisine göre düzelten bir araştırma uygulaması.")

st.sidebar.header("⚙️ Ayarlar")
k_sec = st.sidebar.selectbox(
    "Simetri türü (k)",
    list(K_SECENEKLER),
    format_func=lambda k: K_SECENEKLER[k],
)
esik_modu = st.sidebar.radio(
    "Tesadüfi eşik (B)",
    ["Her görsel için kendi değeri (önerilen)", "Sabit Penrose değeri (0,528)"],
)
sabit_b = esik_modu.startswith("Sabit")
merkez_koyu = st.sidebar.checkbox("Merkezi koyu bölgeye göre ayarla", value=False)
st.sidebar.caption("Negatif kontrol her ölçümde otomatik yapılır (piksel karıştırma, sabit tohum 0).")

# =============================================================================
# GÖRSEL OKUMA VE ÖRNEKLER
# =============================================================================
def gorsel_oku(kaynak):
    """Dosyayı 200×200 gri tonlamaya çevirir. Okunamazsa None döner."""
    try:
        if hasattr(kaynak, "seek"):
            kaynak.seek(0)
        img = Image.open(kaynak).convert("L").resize((IMG_N, IMG_N), Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def ornek_uret(ad):
    img = Image.new("L", (IMG_N, IMG_N), 0)
    d = ImageDraw.Draw(img)
    c = IMG_N // 2
    if ad == "yuvarlak":
        d.ellipse([c - 45, c - 45, c + 45, c + 45], fill=200)
    elif ad == "duzensiz":
        d.ellipse([c - 50, c - 35, c + 25, c + 45], fill=180)
        d.polygon([(c, c), (c + 55, c - 40), (c + 40, c + 20)], fill=150)
    elif ad == "bes_kat":
        for r in range(15, 85, 12):
            pts = []
            for i in range(10):
                a = i * np.pi / 5
                rr = r if i % 2 == 0 else r / 2
                pts.append((c + rr * np.cos(a), c + rr * np.sin(a)))
            d.polygon(pts, outline=255)
    elif ad == "gurultu":
        return np.random.default_rng(42).integers(0, 256, (IMG_N, IMG_N), dtype=np.uint8)
    return np.array(img)


def gorsel_sec(etiket, anahtar, varsayilan):
    """Dosya yükleme ya da örnek seçimi. (görsel dizisi, ad) döndürür."""
    yuklenen = st.file_uploader(f"{etiket}: dosya yükle", type=UPLOAD_TYPES, key=f"up_{anahtar}")
    ornek = st.selectbox(f"{etiket}: veya örnek seç", list(ORNEKLER), index=varsayilan, key=f"ornek_{anahtar}")
    if yuklenen is not None:
        g = gorsel_oku(yuklenen)
        if g is not None:
            return g, yuklenen.name
        st.error("Dosya okunamadı; örnek görsel kullanılıyor.")
    return ornek_uret(ORNEKLER[ornek]), f"Örnek: {ornek}"


# =============================================================================
# ÖLÇÜM ÇEKİRDEĞİ
# =============================================================================
def sim_hesapla(binary, k, radii, cxs, cys):
    """
    Her merkez için S_k = (eşleşen çift) / (geçerli çift) döndürür.
    Çiftler: merkezden r yarıçapında θ açısındaki nokta ile θ + 2π/k açısındaki nokta.
    Şekil: (len(cys), len(cxs)).
    """
    h, w = binary.shape
    aci = np.linspace(0, 2 * np.pi, ACI_SAYISI, endpoint=False)
    R = np.repeat(np.asarray(radii, dtype=float), ACI_SAYISI)
    A = np.tile(aci, len(radii))
    rot = 2 * np.pi / k
    ox1 = np.rint(R * np.cos(A)).astype(int)
    oy1 = np.rint(R * np.sin(A)).astype(int)
    ox2 = np.rint(R * np.cos(A + rot)).astype(int)
    oy2 = np.rint(R * np.sin(A + rot)).astype(int)

    CX, CY = np.meshgrid(np.asarray(cxs), np.asarray(cys))
    CX, CY = CX[..., None], CY[..., None]
    px, py = CX + ox1, CY + oy1
    qx, qy = CX + ox2, CY + oy2

    gecerli = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
    px, qx = np.clip(px, 0, w - 1), np.clip(qx, 0, w - 1)
    py, qy = np.clip(py, 0, h - 1), np.clip(qy, 0, h - 1)

    a = binary[py, px]
    b = binary[qy, qx]
    toplam = gecerli.sum(axis=2)
    eslesen = ((a == b) & gecerli).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(toplam > 0, eslesen / np.maximum(toplam, 1), np.nan)


def merkez_bul(gri, ots, koyu_modu):
    if koyu_modu:
        koyu = gri <= ots
        if 0.02 <= koyu.mean() <= 0.98:
            ys, xs = np.nonzero(koyu)
            return int(round(xs.mean())), int(round(ys.mean()))
    return MERKEZ


def gorsel_analiz(gri, k, sabit, koyu_modu, radii=SKOR_YARICAPLARI, kontrol=True):
    """
    Bir görseli ölçer.
    p: açık piksel oranı, B: şans eşiği, S: eşleşme oranı, U = (S − B)/(1 − B).
    U_ort: yüzeydeki ızgara noktalarının ortalaması.
    U_kontrol: pikseller karıştırılmış görselin U_ort değeri.
    Düz (tek renkli) görselde None döner.
    """
    if gri is None or gri.max() == gri.min():
        return None

    ots = float(gri.mean())
    binary = gri > ots
    p = float(binary.mean())
    B = B_PENROSE if sabit else p ** 2 + (1 - p) ** 2

    xs = np.arange(IZGARA_BASI, IMG_N - IZGARA_BASI, IZGARA_ADIMI)
    S_izg = sim_hesapla(binary, k, radii, xs, xs)
    U_izg = (S_izg - B) / (1 - B)

    mx, my = merkez_bul(gri, ots, koyu_modu)
    S_m = sim_hesapla(binary, k, radii, [mx], [my])[0, 0]

    res = {
        "p": p,
        "B": B,
        "xs": xs,
        "ys": xs,
        "U_izg": U_izg,
        "S_ort": float(np.nanmean(S_izg)),
        "U_ort": float(np.nanmean(U_izg)),
        "U_maks": float(np.nanmax(U_izg)),
        "U_merkez": float((S_m - B) / (1 - B)),
        "U_kontrol": float("nan"),
    }
    if kontrol:
        karisik = np.random.default_rng(0).permutation(gri.ravel()).reshape(gri.shape)
        S_k = sim_hesapla(karisik > ots, k, radii, xs, xs)
        res["U_kontrol"] = float(np.nanmean((S_k - B) / (1 - B)))
    return res


def yorum_etiketi(u):
    if u > YORUM_ESIGI:
        return "tesadüften belirgin yüksek"
    if u < -YORUM_ESIGI:
        return "tesadüften belirgin düşük"
    return "tesadüf beklentisine yakın"


def iki_grup_istatistik(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    fark = m1 - m2
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = fark / sp if sp > 0 else float("nan")

    se2 = v1 / n1 + v2 / n2
    if se2 > 0:
        df_w = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        tk = stats.t.ppf(0.975, df_w)
        se = np.sqrt(se2)
        ci = (fark - tk * se, fark + tk * se)
    else:
        ci = (fark, fark)
    return {"n1": n1, "n2": n2, "fark": fark, "t": t, "p": p, "d": d, "ci": ci}


# =============================================================================
# ÇİZİM YARDIMCILARI
# =============================================================================
def goster(fig):
    st.pyplot(fig)
    plt.close(fig)


def isi_figur(gri, U, xs, ys, baslik):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
    ax.imshow(gri, cmap="gray", alpha=0.45)
    half = IZGARA_ADIMI / 2
    im = ax.imshow(
        U, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0, alpha=0.75, interpolation="nearest",
        extent=[xs[0] - half, xs[-1] + half, ys[-1] + half, ys[0] - half],
    )
    ax.set_xlim(0, IMG_N)
    ax.set_ylim(IMG_N, 0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="U")
    ax.set_title(baslik, fontsize=9)
    ax.axis("off")
    return fig


def fig_kirinim():
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    ax.set_facecolor("#0b0f19")
    for r, n in [(0.2, 10), (0.4, 10), (0.65, 10), (0.9, 20)]:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ax.scatter(r * np.cos(ang), r * np.sin(ang), s=14, c="#00f2fe", alpha=0.9)
    ax.scatter([0], [0], s=30, c="white")
    ax.set_title("Kuasikristal kırınım deseni\n(şematik model)", fontsize=8, color="white")
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def fig_rozet():
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    ang = np.linspace(0, 2 * np.pi, 11)
    r = np.where(np.arange(11) % 2 == 0, 1.0, 0.382)
    x, y = r * np.cos(ang), r * np.sin(ang)
    ax.fill(x, y, color="#ff007f", alpha=0.25)
    ax.plot(x, y, color="#ff007f", lw=1.8)
    ax.set_title("5 katlı şematik rozet\n(ölçekli değil)", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def fig_girih():
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    t = np.linspace(0, 2 * np.pi, 11)[:-1]
    px, py = np.cos(t), np.sin(t)
    for i in range(10):
        j = (i + 3) % 10
        ax.plot([px[i], px[j]], [py[i], py[j]], color="#8b5a2b", lw=1.4)
    ax.scatter(px, py, s=10, c="#8b5a2b")
    ax.set_title("On katlı girih benzeri çizim\n(şematik, ölçekli değil)", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def fig_b_egrisi():
    p = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=110)
    ax.plot(p, p ** 2 + (1 - p) ** 2, color="#1f77b4", label="B(p)")
    ax.axhline(B_PENROSE, color="red", ls="--", label="B_P ≈ 0,528")
    ax.axvline(P_PENROSE, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Açık piksel oranı (p)")
    ax.set_ylabel("Şans eşiği (B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def toplu_grafik(df):
    gruplar = [GRUP_IYI, GRUP_KOTU]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=110)
    for ax, kol, baslik in (
        (axes[0], "U_ort", "Ortalama YSÜM (U_ort)"),
        (axes[1], "U_kontrol", "Negatif kontrol (U_kontrol)"),
    ):
        veri = [df.loc[df["Grup"] == g, kol].to_numpy() for g in gruplar]
        ax.boxplot(veri, widths=0.5, showfliers=False)
        rng = np.random.default_rng(0)
        for i, v in enumerate(veri, start=1):
            ax.scatter(rng.normal(i, 0.04, len(v)), v, s=12, alpha=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(gruplar)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_title(baslik, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# SEKMELER
# =============================================================================
tab_kavram, tab_yan, tab_toplu, tab_isi, tab_sinir = st.tabs([
    "🔭 Kavram", "⚖️ Yan Yana", "📊 Toplu Test", "🌀 Isı Haritası", "⚠️ Sınırlılıklar",
])

# ---------------------------------------------------------------- KAVRAM
with tab_kavram:
    t_pen, t_olc, t_uyg = st.tabs(["📐 Penrose", "🧮 Ölçüm", "🌐 Uygulamalar"])

    with t_pen:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
**Penrose döşemesi (1974):** Roger Penrose, düzlemi boşluk bırakmadan örten ama öteleme simetrisi olmayan (aperiyodik) karo sistemleri tanımladı.

**Altın oran:** Bu sistemlerde kalın ve ince karolar φ ≈ 1,618 ile ilişkili oranlarda bulunur. Kalın karoların toplam karoya oranı 1/φ ≈ 0,618'dir. Bu proje p = 1/φ değerini bir **model varsayımı** olarak kullanır.

**Kuasikristaller:** Dan Shechtman 1982'de düzenli ama periyodik olmayan bir yapı gözlemledi. Bu gözlem için 2011 Nobel Kimya Ödülü verildi. Ödül kuasikristal keşfi içindir; Penrose döşemelerinin ödülü değildir.
""")
        with c2:
            goster(fig_kirinim())
        with c3:
            goster(fig_rozet())
        st.markdown("**Girih:** Lu ve Steinhardt (2007), İslam mimarisindeki girih karo düzenlerinin on katlı Penrose benzeri kuasi-periyodik düzenler içerdiğini gösterdi. Bu, proje bulgusu değildir.")
        goster(fig_girih())

    with t_olc:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**1. Görsel ikileştirme:** Piksel parlaklığı görselin ortalamasından büyükse açık (1), değilse koyu (0) sayılır. p, açık piksel oranıdır.

**2. Şans eşiği B:** İki bağımsız pikselin aynı değeri alma olasılığıdır:
""")
            st.latex(r"B(p)=p^{2}+(1-p)^{2}")
            st.markdown("Pikseller rastgele dağılmış olsaydı beklenen eşleşme oranı budur. Penrose sabiti B_P ≈ 0,528 bir model varsayımıdır.")
            st.markdown("**3. Eşleşme oranı S_k:** Merkez etrafında r yarıçapındaki bir nokta ile 2π/k kadar döndürülmüş noktanın aynı değeri taşıma oranı.")
            st.latex(r"S_k=\dfrac{M_k}{T_k}")
            st.markdown("M_k eşleşen, T_k geçerli çift sayısıdır.")
            st.markdown("**4. Üstünlük skoru U_k:** Eşleşme, şans beklentisinin ne kadar üstünde?")
            st.latex(r"U_k=\dfrac{S_k-B}{1-B}")
            st.markdown("""
- **U ≈ 0:** Eşleşme şans beklentisine yakın.
- **U > 0:** Beklentiden yüksek dönel düzen.
- **U < 0:** Beklentinin altında.

**5. Negatif kontrol:** Aynı görselin pikselleri karıştırılır ve ölçüm tekrarlanır. Gerçek ve kontrol değerleri arasındaki fark, ölçümün piksellerin konumsal düzenine bağlı olduğunu gösterir. Bu bir istatistiksel test değildir.

**Okuma aralığı:** |U| ≤ 0,15 "tesadüf beklentisine yakın" olarak etiketlenir. Bu, okumayı kolaylaştırmak için kullanılan bir proje konvansiyonudur; istatistiksel bir eşik değildir.
""")
        with c2:
            goster(fig_b_egrisi())

    with t_uyg:
        st.markdown("""
- **Dermatoskopi bağlantısı:** ABCDE ölçütünün A (asimetri) öğesi lezyonun eksenlere göre simetrisini değerlendirir. Bu uygulama o değerlendirmenin matematiksel bir analoğunu ölçer. Lezyon segmentasyonu yapmaz ve tanı koymaz.
- **k = 2 bir vekil ölçüdür:** Dermatoskopide kullanılan eksen (yansıma) simetrisi ile 180° nokta simetrisi aynı şey değildir. HAM10000 testinde k = 2 kullanılması bu nedenle vekil bir ölçüdür.
- **Kullanım alanı:** Eğitim, yöntem geliştirme ve araştırma amaçlıdır.
""")

# ---------------------------------------------------------------- YAN YANA
with tab_yan:
    st.markdown("İki görseli yükleyin ya da örnek seçin. İki görsel de aynı ayarlarla (simetri türü, eşik, merkez) ölçülür.")
    c1, c2 = st.columns(2)
    with c1:
        g1, ad1 = gorsel_sec("1. görsel", "a", 0)
    with c2:
        g2, ad2 = gorsel_sec("2. görsel", "b", 1)

    r1 = gorsel_analiz(g1, k_sec, sabit_b, merkez_koyu)
    r2 = gorsel_analiz(g2, k_sec, sabit_b, merkez_koyu)

    if r1 is None or r2 is None:
        st.warning("Tek renkli (düz) bir görsel ölçülemez. Lütfen başka bir görsel seçin.")
    else:
        o1, o2 = st.columns(2)
        with o1:
            goster(isi_figur(g1, r1["U_izg"], r1["xs"], r1["ys"], f"1. görsel · U_ort = {r1['U_ort']:+.3f}"))
        with o2:
            goster(isi_figur(g2, r2["U_izg"], r2["xs"], r2["ys"], f"2. görsel · U_ort = {r2['U_ort']:+.3f}"))

        dU = r2["U_ort"] - r1["U_ort"]
        st.metric("ΔU = U_ort(2. görsel) − U_ort(1. görsel)", f"{dU:+.3f}")

        if dU > 0:
            yon = "2. görsel, 1. görsele göre daha yüksek ortalama dönel düzen gösteriyor."
        elif dU < 0:
            yon = "1. görsel, 2. görsele göre daha yüksek ortalama dönel düzen gösteriyor."
        else:
            yon = "İki görselin ortalama dönel düzeyi aynı çıktı."
        st.info(yon + " Bu bir ölçüm farkıdır; farkın nedenini göstermez.")

        tablo = pd.DataFrame(
            [
                ("Açık alan oranı (p)", f"%{r1['p'] * 100:.1f}", f"%{r2['p'] * 100:.1f}"),
                ("Şans eşiği (B)", f"{r1['B']:.3f}", f"{r2['B']:.3f}"),
                ("Ortalama eşleşme (S_ort)", f"{r1['S_ort']:.3f}", f"{r2['S_ort']:.3f}"),
                ("Ortalama YSÜM (U_ort)", f"{r1['U_ort']:+.3f}", f"{r2['U_ort']:+.3f}"),
                ("Maksimum odak (U_maks)", f"{r1['U_maks']:+.3f}", f"{r2['U_maks']:+.3f}"),
                ("Merkez YSÜM (U_merkez)", f"{r1['U_merkez']:+.3f}", f"{r2['U_merkez']:+.3f}"),
                ("Negatif kontrol (U_kontrol)", f"{r1['U_kontrol']:+.3f}", f"{r2['U_kontrol']:+.3f}"),
                ("Durum (U_ort'a göre)", yorum_etiketi(r1["U_ort"]), yorum_etiketi(r2["U_ort"])),
            ],
            columns=["Ölçüm", "1. görsel", "2. görsel"],
        )
        st.table(tablo.set_index("Ölçüm"))
        st.caption(
            "Negatif kontrol: pikseller rastgele karıştırılıp aynı ölçüm tekrarlanır. "
            "Gerçek ve kontrol değerleri arasındaki fark, ölçümün konumsal düzene bağlı olduğunu gösterir; istatistiksel test değildir."
        )

# ---------------------------------------------------------------- TOPLU TEST
def toplu_calistir(dosyalar_iyi, dosyalar_kotu, k, sabit, koyu, esik_metni):
    satirlar, hatalar = [], []
    hepsi = [(GRUP_IYI, f) for f in dosyalar_iyi] + [(GRUP_KOTU, f) for f in dosyalar_kotu]
    ilerleme = st.progress(0.0, text="Hesaplanıyor…")
    for n, (grup, f) in enumerate(hepsi, start=1):
        g = gorsel_oku(f)
        r = gorsel_analiz(g, k, sabit, koyu) if g is not None else None
        if r is None:
            hatalar.append(f.name)
        else:
            satirlar.append({
                "Grup": grup,
                "Dosya": f.name,
                "p": round(r["p"], 6),
                "B": round(r["B"], 6),
                "S_ort": round(r["S_ort"], 6),
                "U_ort": round(r["U_ort"], 6),
                "U_maks": round(r["U_maks"], 6),
                "U_kontrol": round(r["U_kontrol"], 6),
            })
        ilerleme.progress(n / len(hepsi), text=f"{n}/{len(hepsi)}")
    ilerleme.empty()

    df = pd.DataFrame(satirlar)
    if df.empty or (df["Grup"] == GRUP_IYI).sum() < 2 or (df["Grup"] == GRUP_KOTU).sum() < 2:
        return {"hata": "Okunabilen görsel sayısı her grupta en az 2 olmalı.", "hatalar": hatalar}

    iyi = df.loc[df["Grup"] == GRUP_IYI, "U_ort"]
    kotu = df.loc[df["Grup"] == GRUP_KOTU, "U_ort"]
    return {
        "hata": None,
        "df": df,
        "hatalar": hatalar,
        "k": k,
        "esik": esik_metni,
        "istat": iki_grup_istatistik(iyi, kotu),
    }


def toplu_goster(s):
    if s["hata"]:
        st.error(s["hata"])
        if s["hatalar"]:
            st.warning("Okunamayan veya düz görseller testten çıkarıldı: " + ", ".join(s["hatalar"]))
        return

    df, i = s["df"], s["istat"]
    st.caption(f"Kullanılan simetri türü: k = {s['k']} · Eşik: {s['esik']} · Negatif kontrol: açık")

    ozet = (
        df.groupby("Grup")
        .agg(
            gorsel=("U_ort", "size"),
            s_ort=("S_ort", "mean"),
            u_ort=("U_ort", "mean"),
            u_ss=("U_ort", "std"),
            u_kontrol=("U_kontrol", "mean"),
        )
        .round(4)
        .rename(columns={
            "gorsel": "Görsel sayısı",
            "s_ort": "Ortalama S",
            "u_ort": "Ortalama U_ort",
            "u_ss": "U_ort std. sapma",
            "u_kontrol": "Ortalama U_kontrol",
        })
    )
    st.markdown("### 📋 Grup özeti")
    st.dataframe(ozet)

    m = st.columns(5)
    m[0].metric("ΔU (iyi − kötü)", f"{i['fark']:+.4f}")
    m[1].metric("Welch t", f"{i['t']:.3f}")
    m[2].metric("p (tek yönlü)", f"{i['p']:.4f}")
    m[3].metric("Cohen's d", f"{i['d']:+.3f}")
    m[4].metric("%95 güven aralığı (fark)", f"[{i['ci'][0]:+.3f}, {i['ci'][1]:+.3f}]")

    if np.isfinite(i["p"]) and i["p"] < ALFA:
        st.success(
            f"İyi huylu grubun ortalama U_ort değeri kötü huylu gruptan {i['fark']:+.3f} farklı ve daha yüksek bulundu "
            f"(tek yönlü Welch p = {i['p']:.4f}, α = {ALFA}). Bu sonuç yalnızca bu örneklem ve bu protokol içindir; tanı amacı taşımaz."
        )
    else:
        st.info(
            f"Tek yönlü test α = {ALFA} düzeyinde anlamlı bir fark göstermedi (p = {i['p']:.4f}). "
            "Bu ölçüm, iki grup arasında ayırt edici bir farkı desteklemiyor."
        )

    st.pyplot(toplu_grafik(df))
    plt.close("all")

    st.markdown("### 📄 Görsel bazlı sonuçlar")
    st.dataframe(df)

    if s["hatalar"]:
        st.warning("Okunamayan veya düz görseller testten çıkarıldı: " + ", ".join(s["hatalar"]))

    st.download_button(
        "📥 ysum_sonuclar.csv indir",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ysum_sonuclar.csv",
        mime="text/csv",
    )


with tab_toplu:
    st.markdown(
        "Her grupta en az 2 görsel yükleyin. Rapordaki test **k = 2** ile yapılmıştır. "
        "Kenar çubuğundaki simetri türünü değiştirirseniz sonuçlar o k için hesaplanır."
    )
    c1, c2 = st.columns(2)
    with c1:
        dosya_iyi = st.file_uploader("İyi huylu (nv) görseller", type=UPLOAD_TYPES,
                                     accept_multiple_files=True, key="grp_iyi")
    with c2:
        dosya_kotu = st.file_uploader("Kötü huylu (mel) görseller", type=UPLOAD_TYPES,
                                      accept_multiple_files=True, key="grp_kotu")

    if st.button("🚀 Testi çalıştır", type="primary", key="btn_toplu"):
        if len(dosya_iyi) < 2 or len(dosya_kotu) < 2:
            st.warning("Her iki grupta da en az 2 görsel olmalı.")
        else:
            esik_metni = "sabit B = 0,528" if sabit_b else "her görsel için B(p)"
            st.session_state["toplu"] = toplu_calistir(
                dosya_iyi, dosya_kotu, k_sec, sabit_b, merkez_koyu, esik_metni
            )

    sonuc = st.session_state.get("toplu")
    if sonuc:
        toplu_goster(sonuc)

# ---------------------------------------------------------------- ISI HARİTASI
with tab_isi:
    st.markdown(
        "Seçilen görselin yüzeyindeki her ızgara noktası için U değeri hesaplanır ve ısı haritası olarak çizilir. "
        "Yarıçaplar 10–40 px'dir."
    )
    kh = st.selectbox("Simetri türü (ısı haritası)", list(K_SECENEKLER), index=1,
                      format_func=lambda k: K_SECENEKLER[k], key="isi_k")
    g, ad = gorsel_sec("Görsel", "isi", 2)
    r = gorsel_analiz(g, kh, sabit_b, merkez_koyu, radii=ISI_YARICAPLARI, kontrol=False)

    if r is None:
        st.warning("Tek renkli (düz) bir görsel ölçülemez.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            goster(isi_figur(g, r["U_izg"], r["xs"], r["ys"], f"{ad} · k = {kh}"))
        with c2:
            m = st.columns(3)
            m[0].metric("Açık alan oranı", f"%{r['p'] * 100:.1f}")
            m[1].metric("Merkez YSÜM", f"{r['U_merkez']:+.3f}")
            m[2].metric("Ortalama YSÜM", f"{r['U_ort']:+.3f}")

            U = r["U_izg"]
            flat = np.where(np.isnan(U), -np.inf, U).ravel()
            idx = np.argsort(flat)[::-1][:3]
            st.markdown("**En yüksek üç odak noktası (piksel koordinatı):**")
            for n, fi in enumerate(idx, start=1):
                i_, j_ = np.unravel_index(fi, U.shape)
                st.markdown(f"{n}. (x = {r['xs'][j_]}, y = {r['ys'][i_]}) · U = {U[i_, j_]:+.3f}")

            st.markdown("""
**Renkler:** 🔴 kırmızı, U değerinin yüksek olduğu (beklentiden fazla dönel düzen) alanları; 🔵 mavi, düşük olduğu alanları gösterir.
""")

# ---------------------------------------------------------------- SINIRLILIKLAR
with tab_sinir:
    st.markdown("""
- **Tıbbi değildir:** Tanı, tarama veya karar desteği amacı taşımaz.
- **Ön işleme:** Görseller 200×200 gri tonlamaya çevrilir; en-boy oranı korunmaz.
- **Segmentasyon yoktur:** İkileştirme ortalama parlaklığa göre yapılır. Arka plan da ölçüme dahildir.
- **Merkez:** Varsayılan merkez (100, 100) noktasıdır. İsteğe bağlı koyu bölge merkezi basit bir kuraldır.
- **Negatif kontrol** konumsal düzenin etkisini sınar; istatistiksel test değildir. Hipotez testi yalnızca Toplu Test sekmesinde Welch t-testi ile yapılır.
- **Sonuçlar** yalnızca kullanılan örneklem ve protokol içindir; genellenemez.
- **Yorum aralığı** (±0,15) okuma kolaylığı için kullanılan bir konvansiyondur.
- **HAM10000** verisi yalnızca ticari olmayan kullanım koşullarında kullanılabilir.
""")
    st.error("Bu test klinik araştırma standartlarının altındadır. Sonuç hiçbir şekilde tıbbi geçerlilik taşımaz.")

st.markdown("---")
st.caption("⚠️ Araştırma ve eğitim amaçlıdır. Tıbbi tanı, tarama veya karar için kullanılamaz.")
