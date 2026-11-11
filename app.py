import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd
from scipy import stats

# ==============================================================================
# STREAMLIT SAYFA DÜZENİ VE KOYU TEMA (DARK MODE CSS)
# ==============================================================================
st.set_page_config(
    page_title="YSÜM - Yerel Simetri Üstünlük Modeli (TÜBİTAK 2204-B)",
    page_icon="🌀",
    layout="wide"
)

# Özel CSS ile Arayüz Görseliyle %100 Uyumlu Koyu Tema
st.markdown("""
<style>
    /* Ana Arka Plan ve Yazı Renkleri */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    /* Yan Menü (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #0d131f;
        border-right: 1px solid #1e293b;
    }
    /* Bilgi ve Uyarı Kutuları */
    .stAlert {
        background-color: #111827;
        border: 1px solid #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
    }
    /* Expander ve Kartlar */
    .streamlit-expanderHeader {
        background-color: #111827 !important;
        color: #00f2fe !important;
        border-radius: 6px;
        font-weight: bold;
    }
    /* Tab Butonları */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom-color: #00f2fe !important;
    }
    /* Buton Tasarımı */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #0b0f19;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.5);
        color: #ffffff;
    }
    /* Metrik Kartları */
    [data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌀 YSÜM - Yerel Simetri Üstünlük Modeli Laboratuvarı")
st.caption("Penrose Geometrisi, Endüstriyel Kalite Kontrolü ve Dermatolojik Asimetri Hassas Ölçüm Karar Destek Platformu")

# ------------------------------------------------------------------------------
# TÜBİTAK 2204-B ETİK VE AKADEMİK BİLGİLENDİRME BANTLI ÜST PANEL
# ------------------------------------------------------------------------------
st.info(
    "🔬 **TÜBİTAK 2204-B Araştırma Projesi Çerçevesi:** "
    "Bu yazılım **bir tıbbi tanı veya teşhis cihazı değildir.** "
    "Projemizin temel amacı, periyodik olmayan Penrose kaplamaları ve geometrik desenlerde geliştirilen "
    "**Yerel Simetri Üstünlük Modeli (YSÜM)** matematiksel altyapısını; hem **endüstriyel yüzey kalite kontrolünde** "
    "hem de **dermatolojik lezyon sınırlarının (ABCDE 'A' Kriteri) matematiksel asimetri risk derecelendirmesinde** "
    "hassas ve nicel bir Karar Destek Sistemi olarak ampirik şekilde sınamaktır."
)

# ------------------------------------------------------------------------------
# DETAYLI DİSİPLİNLERARASI TEORİ VE BİLİMSEL LİTERATÜR BÖLÜMÜ (5 SEKMELİ ZENGİN YAPI)
# ------------------------------------------------------------------------------
with st.expander("📌 Penrose Geometrisi, Endüstriyel Yüzey Kontrolü ve Dermatolojik Asimetri Teorisi", expanded=True):
    tab_teori1, tab_teori2, tab_teori3, tab_teori4, tab_teori5 = st.tabs([
        "📐 Penrose & Kuasikristaller",
        "🧮 YSÜM ($U_k$) Matematiği",
        "🏭 Endüstriyel & Sanayi Uygulamaları",
        "🩺 Dermatolojik Asimetri & Risk Ölçümü",
        "🕌 Girih Sanatı & Lu-Steinhardt (2007)"
    ])

    with tab_teori1:
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown(r"""
            ### 📐 Penrose Kaplamaları ve Altın Oran
            * **Periyodik Olmayan Yapı (Aperiodic Tiling):** Roger Penrose tarafından 1974'te keşfedilen bu yapılar, düzlemi boşluksuz örter ancak öteleme (kaydırma) simetrisine sahip değildir.
            * **Altın Oran ($\Phi \approx 1.618$):** Karo alan oranları ve dizilimi Altın Oran'a ve Fibonacci sayılarına yakınsar ($p = 1/\Phi \approx 0.618$).
            * **2011 Nobel Kimya Ödülü Bağlantısı:** Dan Shechtman 1982'de kuasikristalleri keşfetmiş ve 2011 Nobel Kimya Ödülü'nü kazanmıştır. Penrose kaplamaları bu fiziksel kuasikristal yapının teorik altyapısını oluşturur.
            """)
        with col_t2:
            fig_n, ax_n = plt.subplots(figsize=(3, 3), dpi=100)
            fig_n.patch.set_facecolor("#0b0f19")
            ax_n.set_facecolor("#0b0f19")
            for ring_r in [0.2, 0.4, 0.65, 0.9]:
                n_dots = 10 if ring_r < 0.8 else 20
                angles_n = np.linspace(0, 2*np.pi, n_dots, endpoint=False)
                ax_n.scatter(ring_r * np.cos(angles_n), ring_r * np.sin(angles_n), c="#00f2fe", s=15, alpha=0.9)
            ax_n.scatter(0, 0, c="white", s=30)
            ax_n.set_title("Shechtman Kırınım Deseni (Model)", fontsize=8, color="white", pad=4)
            ax_n.axis("off")
            st.pyplot(fig_n)
            plt.close(fig_n)
        with col_t3:
            fig_p, ax_p = plt.subplots(figsize=(3, 3), dpi=100)
            fig_p.patch.set_facecolor("#0b0f19")
            ax_p.set_facecolor("#0b0f19")
            angles_p = np.linspace(0, 2*np.pi, 11)
            r_mesh = np.where(np.arange(11) % 2 == 0, 1.0, 0.382)
            ax_p.plot(r_mesh * np.cos(angles_p), r_mesh * np.sin(angles_p), color="#ff007f", lw=1.8)
            ax_p.fill(r_mesh * np.cos(angles_p), r_mesh * np.sin(angles_p), color="#ff007f", alpha=0.25)
            ax_p.set_title("5-Katlı Geometrik Şema", fontsize=8, color="#00f2fe", pad=4)
            ax_p.axis("off")
            st.pyplot(fig_p)
            plt.close(fig_p)

    with tab_teori2:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(r"""
            ### 🧮 YSÜM ($U_k$) ve Şans Arındırması Formülü
            * **İkileştirilmiş Eşleşme ($S_k$):** Binarize görsel matrisi ($I_{bin}$) üzerinde merkez etrafında $2\pi/k$ dönüşlerindeki piksel çakışma oranıdır.
            * **Dinamik Şans Eşiği ($B$):** Açık piksel oranı $p$ için rastgele eşleşme ihtimali:
              $$B(p) = p^2 + (1-p)^2$$
            * **Penrose Cebirsel Şans Eşiği ($B_{Penrose}$):** Altın oran payı $p = 1/\Phi \approx 0.618$ için $B \approx \mathbf{0.528}$.
            * **Üstünlük Skoru ($U_k$):**
              $$U_k = \frac{S_k - B}{1.0 - B}$$
            """)
        with col_m2:
            fig_b, ax_b = plt.subplots(figsize=(3.5, 2.5), dpi=100)
            fig_b.patch.set_facecolor("#0b0f19")
            ax_b.set_facecolor("#0d131f")
            p_vals = np.linspace(0, 1, 100)
            b_vals = p_vals**2 + (1 - p_vals)**2
            ax_b.plot(p_vals, b_vals, color="#00f2fe", label=r"$B(p) = p^2+(1-p)^2$")
            ax_b.axhline(0.528, color="#ff007f", linestyle="--", label=r"Penrose B $\approx 0.528$")
            ax_b.set_xlabel("Açık Piksel Oranı (p)", fontsize=8, color="white")
            ax_b.set_ylabel("Şans Eşiği (B)", fontsize=8, color="white")
            ax_b.tick_params(colors="white", labelsize=7)
            ax_b.legend(fontsize=7, facecolor="#111827", edgecolor="#1e293b", labelcolor="white")
            ax_b.grid(True, alpha=0.2, color="#334155")
            st.pyplot(fig_b)
            plt.close(fig_b)

    with tab_teori3:
        st.markdown("""
        ### 🏭 Endüstriyel & Sanayi Uygulamaları (Yüzey Kalite Kontrolü)
        """)
        col_app1, col_app2, col_app3 = st.columns(3)
        with col_app1:
            st.success(
                "🔬 **1. Malzeme Bilimi & İnce Filmler**\n\n"
                "* **Kuasikristal Kaplamalar:** Aşınmaz ve yapışmaz yüzeylerin dönel simetri ve homojenlik derecesinin nicel ölçümü.\n"
                "* **Yüzey Anizotropisi:** Mikroskop görüntülerinde atomik/kristal dizilim yönelim analizi."
            )
        with col_app2:
            st.info(
                "⚡ **2. Fotonik & Optik Izgaralar**\n\n"
                "* **Işık Hapsedici Yapılar:** Fotonik kristal matrislerinde periyodik olmayan dönel düzenlilik bozulmalarının haritalanması.\n"
                "* **Fiber Optik Üretimi:** Izgara simetrisi hatalarının tespiti."
            )
        with col_app3:
            st.warning(
                "🏭 **3. Fabrika Üretim Kalite Kontrolü**\n\n"
                "* **Tekstil & Seramik Otomasyonu:** Dokuma, çini ve ambalaj yüzeylerindeki simetri kırılmalarının ve kılcal çatlakların hassas tespiti.\n"
                "* **Ürün Sınıflandırması:** YSÜM skoru ($U_k$) ile otomatik PASS/FAIL kalite barajı."
            )

    with tab_teori4:
        col_med1, col_med2 = st.columns(2)
        with col_med1:
            st.markdown(r"""
            ### 🩺 Dermatolojik Asimetri (ABCDE "A" Kriteri) ve Hassas Risk Ölçümü
            * **Dermatolojik Asimetri (A - Asymmetry):** Ben ve lezyon incelemelerinde kötü huylu (melanom) riski taşıyan lezyonların en belirgin özelliği sınırların iki eksende asimetrik olmasıdır.
            * **YSÜM $k=2$ Hassas Ölçüm Modeli:** Lezyon sınırlarının 2-katlı dönel simetriden sapmasını ölçerek nicel bir asimetri derecesi sunar.
            * **Risk Sınıflandırması Eğilimi:**
              * 🟢 **DÜŞÜK ASİMETRİ RİSKİ ($U_2 \ge 0.45$):** Düzenli, simetrik ve iyi huylu eğilimli lezyon biçimi.
              * 🟡 **ORTA ASİMETRİ RİSKİ ($0.15 \le U_2 < 0.45$):** Hafif sınır asimetrisi; düzenli dermatolojik takip önerilir.
              * 🔴 **YÜKSEK ASİMETRİ RİSKİ ($U_2 < 0.15$):** Belirgin sınır bozulması; şüpheli / kötü huylu eğilimli asimetrik biçim.
            """)
        with col_med2:
            fig_m, ax_m = plt.subplots(figsize=(3, 3), dpi=100)
            fig_m.patch.set_facecolor("#0b0f19")
            ax_m.set_facecolor("#0b0f19")
            theta = np.linspace(0, 2*np.pi, 200)
            r_sym = 0.8 + 0.05 * np.cos(5*theta)
            r_asym = 0.8 + 0.25 * np.sin(3*theta) + 0.1 * np.cos(7*theta)
            ax_m.plot(r_sym * np.cos(theta), r_sym * np.sin(theta), color="#00f2fe", linestyle="--", label="İyi Huylu Eğilimli (Simetrik)", lw=1.5)
            ax_m.plot(r_asym * np.cos(theta), r_asym * np.sin(theta), color="#ff007f", linestyle="-", label="Şüpheli / Kötü Huylu Eğilimli", lw=1.8)
            ax_m.set_title("ABCDE 'A' Asimetri Şeması", fontsize=8, color="white", pad=4)
            ax_m.legend(fontsize=7, loc="lower right", facecolor="#111827", labelcolor="white")
            ax_m.axis("off")
            st.pyplot(fig_m)
            plt.close(fig_m)

    with tab_teori5:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            ### 🕌 Girih Sanatı & Lu-Steinhardt (2007) Bulguları
            * **Akademik Literatür:** Peter J. Lu ve Paul J. Steinhardt (2007, *Science*), 15. yüzyıl İslam mimarisindeki (ör. İsfahan Darb-ı İmam) Girih karolarının periyodik olmayan Penrose benzeri geometrik düzenler içerdiğini göstermiştir.
            * **Restorasyon Rehberi:** Tarihi yapılarda eksik veya yıpranmış çini motiflerinin açısal simetri merkezlerinin sayısal olarak haritalanması.
            """)
        with col_g2:
            fig_g, ax_g = plt.subplots(figsize=(3, 3), dpi=100)
            fig_g.patch.set_facecolor("#0b0f19")
            ax_g.set_facecolor("#0b0f19")
            angles_g = np.linspace(0, 2*np.pi, 21)
            r_girih = np.where(np.arange(21) % 2 == 0, 1.0, 0.45)
            ax_g.plot(r_girih * np.cos(angles_g), r_girih * np.sin(angles_g), color="#00f2fe", lw=2)
            ax_g.fill(r_girih * np.cos(angles_g), r_girih * np.sin(angles_g), color="#00f2fe", alpha=0.25)
            ax_g.set_title("10-Katlı Girih Çizim Modeli", fontsize=8, color="white", pad=4)
            ax_g.axis("off")
            st.pyplot(fig_g)
            plt.close(fig_g)

# ------------------------------------------------------------------------------
# YAN MENÜ (AYARLAR VE ÇALIŞMA MODLARI)
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Çalışma Modu ve Ayarlar")

analiz_modu = st.sidebar.radio(
    "📌 Çalışma Modu Seçin:",
    [
        "🖼️ Tekli Görsel Analizi & Hassas Simetri Haritası",
        "🩺 İkili Dermatolojik Karşılaştırma Modu (Yan Yana Takip)",
        "📊 Toplu Görsel Analizi & İstatistiksel Hipotez Testi (Welch's t-Testi)"
    ]
)

k_secim = st.sidebar.selectbox(
    "Simetri Kat Sayısı (k):",
    ["Hepsini Otomatik Tara", "k=2 (İki Eksenli / Dermatoloji)", "k=5 (Penrose/Beşgen)", "k=6 (Altıgen)", "k=8 (Sekizgen)"]
)

b_modu = st.sidebar.radio(
    "Şans Eşiği (B) Modu:",
    ["Otomatik (Görsel Piksellerinden B(p))", "Cebirsel Penrose Eşiği (B = 0.528)", "Manuel Eşik"]
)

if "Manuel" in b_modu:
    b_esik_val = st.sidebar.slider("Manuel Şans Eşiği (B):", 0.10, 0.90, 0.528, 0.01)
elif "Cebirsel" in b_modu:
    b_esik_val = 0.528
else:
    b_esik_val = None

roi_filtre = st.sidebar.checkbox("🎯 Nesne / Maske Odaklı Tarama (Arka Planı Muaf Tut)", value=True)
negatif_kontrol = st.sidebar.checkbox("Negatif Kontrol (Tekrarlanabilir Seed=42 Shuffling)", value=True)

# ------------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------------------------
def ornek_desen_olustur(tip, seed_val=42):
    np.random.seed(seed_val)
    w, h = 200, 200
    img_pil = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img_pil)
    cx, cy = w // 2, h // 2
    
    if "Düzgün Sınırlı Model" in tip or "İyi Huylu" in tip or "Grup A" in tip:
        var = np.random.randint(-3, 4)
        draw.ellipse([cx - 45 + var, cy - 45 - var, cx + 45 + var, cy + 45 - var], fill=200, outline=255)
    elif "Düzensiz Sınırlı Model" in tip or "Kötü Huylu" in tip or "Grup B" in tip:
        p1 = np.random.randint(40, 60)
        p2 = np.random.randint(30, 50)
        draw.ellipse([cx - p1, cy - p2, cx + 25, cy + 45], fill=180, outline=255)
        draw.polygon([(cx, cy), (cx + 55, cy - 40), (cx + 40, cy + 20)], fill=150)
    elif "5-Katlı Yıldız" in tip:
        for r in range(15, 85, 12):
            points = []
            for i in range(10):
                angle = i * np.pi / 5
                radius = r if i % 2 == 0 else r // 2
                x = cx + int(radius * np.cos(angle))
                y = cy + int(radius * np.sin(angle))
                points.append((x, y))
            draw.polygon(points, outline=255, fill=120 if r % 24 == 0 else 0)
    elif "8-Katlı Girih" in tip:
        for r in range(20, 90, 15):
            points = []
            for i in range(16):
                angle = i * np.pi / 8
                radius = r if i % 2 == 0 else r // 2.2
                x = cx + int(radius * np.cos(angle))
                y = cy + int(radius * np.sin(angle))
                points.append((x, y))
            draw.polygon(points, outline=255, fill=180 if r % 30 == 0 else 0)
    elif "Rastgele Gürültü" in tip:
        arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        return arr
        
    return np.array(img_pil)

def hesapla_ysum_matrisi(resim_np, k, b_degeri, ots_thresh, yaricaplar=None, ornek_sayisi=36, adim=8, maske_odakli=True):
    if yaricaplar is None:
        yaricaplar = [8, 16, 24]
    
    binary_np = (resim_np > ots_thresh).astype(np.uint8)
    yukseklik, genislik = binary_np.shape
    
    x_grid = np.arange(10, genislik - 10, adim)
    y_grid = np.arange(10, yukseklik - 10, adim)
    u_matrisi = np.zeros((len(y_grid), len(x_grid)))
    donme_acisi = 2 * np.pi / k

    for i, cy in enumerate(y_grid):
        for j, cx in enumerate(x_grid):
            if maske_odakli and binary_np[cy, cx] == 0:
                u_matrisi[i, j] = 0.0
                continue

            toplam_nokta, eslesen_nokta = 0, 0
            for r in yaricaplar:
                acilar = np.linspace(0, 2 * np.pi, ornek_sayisi, endpoint=False)
                for aci in acilar:
                    px = int(round(cx + r * np.cos(aci)))
                    py = int(round(cy + r * np.sin(aci)))
                    dpx = int(round(cx + r * np.cos(aci + donme_acisi)))
                    dpy = int(round(cy + r * np.sin(aci + donme_acisi)))
                    
                    if (0 <= px < genislik and 0 <= py < yukseklik and 0 <= dpx < genislik and 0 <= dpy < yukseklik):
                        if not maske_odakli or (binary_np[py, px] == 1 or binary_np[dpy, dpx] == 1):
                            toplam_nokta += 1
                            if binary_np[py, px] == binary_np[dpy, dpx]:
                                eslesen_nokta += 1
            if toplam_nokta > 0:
                sk = eslesen_nokta / toplam_nokta
                u_matrisi[i, j] = (sk - b_degeri) / (1.0 - b_degeri) if b_degeri < 1.0 else 0.0
    return u_matrisi, genislik, yukseklik

# ==============================================================================
# MOD 1: TEKLİ GÖRSEL ANALİZİ & HASSAS SİMETRİ HARİTASI
# ==============================================================================
if "Tekli" in analiz_modu:
    st.markdown("---")
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        uploaded_file = st.file_uploader("📂 Analiz edilecek resmi yükleyin (PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])

    with col_input2:
        st.markdown("**Veya Örnek Sentetik Model Seçin:**")
        ornek_tipi = st.selectbox("Sentetik Test Modeli Seçin:", [
            "Yüklenen Resmi Kullan", 
            "🩺 Sentetik Model 1: Düzgün Sınırlı Model (İyi Huylu Eğilimli)",
            "🩺 Sentetik Model 2: Düzensiz Sınırlı Model (Şüpheli / Kötü Huylu Eğilimli)",
            "📐 Sentetik Model 3: 5-Katlı Penrose Yıldız Desen", 
            "🕌 Sentetik Model 4: 8-Katlı Girih Motifi", 
            "🎲 Sentetik Model 5: Rastgele Gürültü"
        ])

    resim_np = None
    if uploaded_file is not None and ornek_tipi == "Yüklenen Resmi Kullan":
        img = Image.open(uploaded_file).convert("L")
        img.thumbnail((250, 250))
        resim_np = np.array(img)
    elif ornek_tipi != "Yüklenen Resmi Kullan":
        resim_np = ornek_desen_olustur(ornek_tipi)

    if resim_np is not None:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(resim_np, caption="Analiz Edilen Görsel Matrisi", width=180, clamp=True)

        with col_img2:
            ots_thresh = int(np.mean(resim_np))
            if ots_thresh == 0: ots_thresh = 128
            
            binary_np = (resim_np > ots_thresh).astype(np.uint8)
            p = float(np.mean(binary_np))
            b_kullanilan = (p**2) + ((1.0 - p)**2) if b_esik_val is None else b_esik_val

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("İkileştirme Eşiği", f"{ots_thresh}")
            m_col2.metric("Piksel Oranı (p)", f"%{p*100:.1f}")
            m_col3.metric("Hesaplanan B Eşiği", f"{b_kullanilan:.3f}")

        if st.button("🚀 YSÜM Analizini Başlat", type="primary"):
            st.markdown("---")
            st.subheader("🖼️ Görsel Tabanlı Dönel Simetri Isı Haritası Analizi")

            if "2" in k_secim: k_listesi = [2]
            elif "5" in k_secim: k_listesi = [5]
            elif "6" in k_secim: k_listesi = [6]
            elif "8" in k_secim: k_listesi = [8]
            else: k_listesi = [2, 5, 6, 8]

            cols = st.columns(len(k_listesi))
            skorlar, maks_skorlar, top_coords_dict = {}, {}, {}

            for idx, k in enumerate(k_listesi):
                with cols[idx]:
                    u_mat, g, y = hesapla_ysum_matrisi(resim_np, k, b_kullanilan, ots_thresh, maske_odakli=roi_filtre)
                    max_u = float(np.max(u_mat))
                    
                    non_zero = u_mat[u_mat > 0.0]
                    mean_u = float(np.mean(non_zero)) if len(non_zero) > 0 else float(np.mean(u_mat))
                    
                    skorlar[k] = mean_u
                    maks_skorlar[k] = max_u

                    flat_indices = np.argsort(u_mat.ravel())[::-1][:3]
                    coords = []
                    for fi in flat_indices:
                        row_i, col_j = np.unravel_index(fi, u_mat.shape)
                        coords.append((10 + col_j * 8, 10 + row_i * 8, float(u_mat[row_i, col_j])))
                    top_coords_dict[k] = coords

                    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                    fig.patch.set_facecolor("#0b0f19")
                    ax.imshow(resim_np, cmap="gray", alpha=0.4)
                    im = ax.imshow(u_mat, cmap="RdYlBu_r", extent=[10, g - 10, y - 10, 10], alpha=0.75, vmin=-0.5, vmax=1.0)
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(colors="white", labelsize=7)
                    cbar.set_label("U_k", color="white", fontsize=8)
                    ax.set_title(f"k={k} Katlı Simetri\n(Maks={max_u:.2f}, Nesne Ort={mean_u:.2f})", fontsize=9, fontweight="bold", color="white")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

            baskin_k = max(skorlar, key=skorlar.get)
            baskin_ort = skorlar[baskin_k]
            baskin_maks = maks_skorlar[baskin_k]
            st.success(f"🏆 **Analiz Bulgusu:** En yüksek ortalama skora sahip yapı **{baskin_k}-Katlı Simetridir** (Ortalama U: **{baskin_ort:.2f}**, Maks: **{baskin_maks:.2f}**)")

            # DETAYLI ANALİZ VE DEĞERLENDİRME REHBERİ
            st.markdown("---")
            st.subheader("💡 Detaylı Analiz & Sonuçlar Ne Anlama Geliyor?")
            
            rep_tab1, rep_tab2, rep_tab3, rep_tab4, rep_tab5 = st.tabs([
                "📊 Sayısal Metrikler & Kontrol",
                "🔍 Odak Koordinatları",
                "🩺 Dermatolojik Risk Sınıflandırması",
                "🏭 Fabrika Kalite Değerlendirmesi",
                "📖 Sonuçları Okuma Rehberi"
            ])
            neg_ort = 0.0
            if negatif_kontrol:
                np.random.seed(42)
                shuffled_test = resim_np.copy().flatten()
                np.random.shuffle(shuffled_test)
                u_mat_test, _, _ = hesapla_ysum_matrisi(shuffled_test.reshape(resim_np.shape), baskin_k, b_kullanilan, ots_thresh, maske_odakli=False)
                neg_ort = float(np.mean(u_mat_test))
            
            fark_u = baskin_ort - neg_ort

            with rep_tab1:
                st.markdown(
                    f"### 📊 Elde Edilen Metrik Verileri\n"
                    f"* **Baskın Dönel Simetri:** **{baskin_k}-Katlı**\n"
                    f"* **Ortalama YSÜM Skoru ($U_{{ort}}$):** **{baskin_ort:.3f}**\n"
                    f"* **Maksimum Odak Skoru ($U_{{maks}}$):** **{baskin_maks:.3f}**\n"
                    f"* **Negatif Kontrol Skoru ($U_{{kontrol}}$):** **{neg_ort:.3f}** (Piksel Shuffling)\n"
                    f"* **Göreli Sinyal Farkı ($\Delta U$):** **+{fark_u:.3f}** (Sinyal / Gürültü Ayrımı)"
                )
            with rep_tab2:
                st.markdown(
                    f"### 🔍 En Yüksek 3 Simetri Odak Noktası ({baskin_k}-Katlı)\n"
                    f"1. **Birincil Odak:** `(X: {top_coords_dict[baskin_k][0][0]} px, Y: {top_coords_dict[baskin_k][0][1]} px)` — YSÜM Skoru: **{top_coords_dict[baskin_k][0][2]:.2f}**\n"
                    f"2. **İkincil Odak:** `(X: {top_coords_dict[baskin_k][1][0]} px, Y: {top_coords_dict[baskin_k][1][1]} px)` — YSÜM Skoru: **{top_coords_dict[baskin_k][1][2]:.2f}**\n"
                    f"3. **Üçüncül Odak:** `(X: {top_coords_dict[baskin_k][2][0]} px, Y: {top_coords_dict[baskin_k][2][1]} px)` — YSÜM Skoru: **{top_coords_dict[baskin_k][2][2]:.2f}**"
                )
            with rep_tab3:
                k2_skor = skorlar.get(2, baskin_ort)
                st.markdown("### 🩺 Dermatolojik Asimetri (ABCDE 'A' Kriteri) Hassas Risk Derecelendirmesi")
                if k2_skor >= 0.45:
                    st.success(
                        f"🟢 **DÜŞÜK ASİMETRİ RİSKİ ($U_2 = {k2_skor:.3f} \\ge 0.45$) — İYİ HUYLU EĞİLİMLİ BİÇİM**\n\n"
                        "Lezyon sınırları iki eksende yüksek dönel simetrik homojenlik göstermektedir. "
                        "Şans arındırılmış modelde asimetrik bozulma tespit edilmemiştir."
                    )
                elif k2_skor >= 0.15:
                    st.warning(
                        f"🟡 **ORTA DERECE ASİMETRİ RİSKİ (${0.15:.2f} \\le U_2 = {k2_skor:.3f} < 0.45$) — ŞÜPHELİ / TAKİP EDİLMESİ ÖNERİLEN BİÇİM**\n\n"
                        "Lezyon sınırlarında kısmi açısal simetri kayıpları mevcuttur. "
                        "Dermatolojik takip ve ABCDE 'A' kriteri kontrolü önerilir."
                    )
                else:
                    st.error(
                        f"🔴 **YÜKSEK ASİMETRİ RİSKİ ($U_2 = {k2_skor:.3f} < 0.15$) — ŞÜPHELİ / KÖTÜ HUYLU EĞİLİMLİ ASİMETRİK BİÇİM**\n\n"
                        "Lezyon sınırları simetriden belirgin şekilde sapmaktadır. "
                        "Açısal asimetri skoru kritik eşiğin altındadır; klinik inceleme uyarısı verilir."
                    )
            with rep_tab4:
                st.markdown("### 🏭 Fabrika Yüzey Kalite Kontrolü Değerlendirmesi")
                if baskin_ort >= 0.40:
                    st.success(f"🟢 **FABRİKA OTOMASYON KARARI: ONAYLANDI (PASS) ($U_{{ort}} = {baskin_ort:.3f} \\ge 0.40$)**\n\nYüzey dokusu homojen ve simetriktir.")
                elif baskin_ort >= 0.15:
                    st.warning(f"🟡 **FABRİKA OTOMASYON KARARI: İNCELEME GEREKLİ (MARGINAL) (${0.15:.2f} \\le U_{{ort}} < 0.40$)**\n\nYüzeyde kısmi doku düzensizliği.")
                else:
                    st.error(f"🔴 **FABRİKA OTOMASYON KARARI: HATALI ÜRÜN (FAIL) ($U_{{ort}} < 0.15$)**\n\nYüzeyde çatlak, leke veya asimetrik bozulma tespit edildi.")

            with rep_tab5:
                st.markdown(r"""
                ### 📖 Isı Haritası ve Skor Yorumlama Rehberi
                
                #### 1. YSÜM Skoru ($U_k$) Değerleri Ne Anlama Gelir?
                * **$U_k \ge 0.50$ (Mükemmel / Güçlü Simetri):** Görselde şans beklentisinin çok üzerinde, belirgin bir geometrik düzen vardır.
                * **$0.20 \le U_k < 0.50$ (Kısmi / Orta Simetri):** Görselin belirli bölgelerinde lokal simetri odakları mevcuttur.
                * **$0.00 \le U_k < 0.20$ (Zayıf Simetri / Şans Seviyesi):** Belirgin dönel simetri yoktur; değer rastgele piksel eşleşmesine yakındır.
                * **$U_k < 0.00$ (Asimetrik Yapı):** Piksel dizilimi rastgele şansın bile altındadır, asimetrik bir görünüm sunar.

                #### 2. Isı Haritasındaki Renkler Nasıl Okunur?
                * 🔴 **Kırmızı / Turuncu Alanlar:** YSÜM skorunun en yüksek olduğu **simetri merkez odaklarını** temsil eder.
                * 🟡 **Sarı / Yeşil Alanlar:** Geçiş bölgeleri ve orta seviye düzenlilik.
                * 🔵 **Mavi / Koyu Mavi Alanlar:** Düzensiz, asimetrik veya gürültülü alanlar.

                #### 3. Negatif Kontrol ve Fark ($\Delta U$) Neyi Kanıtlar?
                * Orijinal resimdeki pikseller rastgele karıştırıldığında (shuffling) skor $U \approx 0.00$'a düşüyorsa, görseldeki simetri **tesadüf değil, matematiksel bir tasarımdır**.
                """)

# ==============================================================================
# MOD 2: İKİLİ GÖRSEL KARŞILAŞTIRMA MODU (YAN YANA TAKİP)
# ==============================================================================
elif "İkili" in analiz_modu:
    st.markdown("---")
    st.subheader("🩺 İkili Dermatolojik Karşılaştırma Paneli (Yan Yana Takip)")
    st.caption("İki farklı görseli veya muayene numunesini ikili piksel matrisi seviyesinde yan yana karşılaştırarak asimetri ve düzenlilik farkını ($\Delta U$) analiz edin.")

    col_cmp1, col_cmp2 = st.columns(2)

    with col_cmp1:
        st.markdown("### 🖼️ 1. Görsel (Referans Muayene / Normal Ben)")
        up_file1 = st.file_uploader("1. Resmi Yükleyin", type=["png", "jpg", "jpeg", "webp"], key="up1")
        sel_ornek1 = st.selectbox("Veya 1. Sentetik Modeli Seçin:", [
            "🩺 Sentetik Model 1: Düzgün Sınırlı Model (İyi Huylu Eğilimli)",
            "🩺 Sentetik Model 2: Düzensiz Sınırlı Model (Şüpheli / Kötü Huylu Eğilimli)",
            "📐 Sentetik Model 3: 5-Katlı Yıldız Desen",
            "🎲 Sentetik Model 5: Rastgele Gürültü"
        ], index=0, key="sel1")

    with col_cmp2:
        st.markdown("### 🖼️ 2. Görsel (Takip Edilen Muayene / Şüpheli Lezyon)")
        up_file2 = st.file_uploader("2. Resmi Yükleyin", type=["png", "jpg", "jpeg", "webp"], key="up2")
        sel_ornek2 = st.selectbox("Veya 2. Sentetik Modeli Seçin:", [
            "🩺 Sentetik Model 2: Düzensiz Sınırlı Model (Şüpheli / Kötü Huylu Eğilimli)",
            "🩺 Sentetik Model 1: Düzgün Sınırlı Model (İyi Huylu Eğilimli)",
            "📐 Sentetik Model 3: 5-Katlı Yıldız Desen",
            "🎲 Sentetik Model 5: Rastgele Gürültü"
        ], index=0, key="sel2")

    resim1_np = np.array(Image.open(up_file1).convert("L").resize((200, 200))) if up_file1 is not None else ornek_desen_olustur(sel_ornek1)
    resim2_np = np.array(Image.open(up_file2).convert("L").resize((200, 200))) if up_file2 is not None else ornek_desen_olustur(sel_ornek2)

    col_v1, col_v2 = st.columns(2)
    with col_v1: st.image(resim1_np, caption="1. Görsel Matrisi", width=180, clamp=True)
    with col_v2: st.image(resim2_np, caption="2. Görsel Matrisi", width=180, clamp=True)

    if st.button("🚀 İkili Karşılaştırma Analizini Başlat", type="primary"):
        k_val = 2 if "2" in k_secim else (5 if "5" in k_secim else (6 if "6" in k_secim else 8))

        ots1 = int(np.mean(resim1_np)) or 128
        p1 = float(np.mean(resim1_np > ots1))
        b1 = (p1**2) + ((1.0 - p1)**2) if b_esik_val is None else b_esik_val
        u_mat1, g1, y1 = hesapla_ysum_matrisi(resim1_np, k_val, b1, ots1, maske_odakli=roi_filtre)
        non_zero1 = u_mat1[u_mat1 > 0.0]
        mean_u1 = float(np.mean(non_zero1)) if len(non_zero1) > 0 else float(np.mean(u_mat1))
        max_u1 = float(np.max(u_mat1))

        ots2 = int(np.mean(resim2_np)) or 128
        p2 = float(np.mean(resim2_np > ots2))
        b2 = (p2**2) + ((1.0 - p2)**2) if b_esik_val is None else b_esik_val
        u_mat2, g2, y2 = hesapla_ysum_matrisi(resim2_np, k_val, b2, ots2, maske_odakli=roi_filtre)
        non_zero2 = u_mat2[u_mat2 > 0.0]
        mean_u2 = float(np.mean(non_zero2)) if len(non_zero2) > 0 else float(np.mean(u_mat2))
        max_u2 = float(np.max(u_mat2))

        out_col1, out_col2 = st.columns(2)
        with out_col1:
            fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=100)
            fig1.patch.set_facecolor("#0b0f19")
            ax1.imshow(resim1_np, cmap="gray", alpha=0.4)
            im1 = ax1.imshow(u_mat1, cmap="RdYlBu_r", extent=[10, g1 - 10, y1 - 10, 10], alpha=0.75, vmin=-0.5, vmax=1.0)
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(colors="white", labelsize=7)
            ax1.set_title(f"1. Görsel (Ort U: {mean_u1:.2f})", fontsize=9, fontweight="bold", color="white")
            ax1.axis("off")
            st.pyplot(fig1)
            plt.close(fig1)

        with out_col2:
            fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=100)
            fig2.patch.set_facecolor("#0b0f19")
            ax2.imshow(resim2_np, cmap="gray", alpha=0.4)
            im2 = ax2.imshow(u_mat2, cmap="RdYlBu_r", extent=[10, g2 - 10, y2 - 10, 10], alpha=0.75, vmin=-0.5, vmax=1.0)
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(colors="white", labelsize=7)
            ax2.set_title(f"2. Görsel (Ort U: {mean_u2:.2f})", fontsize=9, fontweight="bold", color="white")
            ax2.axis("off")
            st.pyplot(fig2)
            plt.close(fig2)

        delta_u = mean_u2 - mean_u1
        st.metric(label="Skor Değişim Farkı (ΔU = U_ort2 - U_ort1)", value=f"{delta_u:+.3f}")

        # DETAYLI İKİLİ KARŞILAŞTIRMA RAPORU VE OKUMA REHBERİ
        st.markdown("---")
        st.subheader("💡 İkili Karşılaştırma Analizi & Sonuçlar Nasıl Okunur?")

        tab_cmp_a, tab_cmp_b, tab_cmp_c = st.tabs([
            "📊 Karşılaştırmalı Metrik Tablosu",
            "🔍 Net Fark & Odak Değişimi",
            "📖 Karşılaştırma Okuma Rehberi"
        ])

        with tab_cmp_a:
            st.table({
                "Parametre Metriği": [
                    "İkileştirme Eşiği (Otsu)",
                    "Açık Piksel Payı (p)",
                    "Hesaplanan Şans Eşiği (B)",
                    "Ortalama YSÜM Skoru (U_ort)",
                    "Maksimum Odak Skoru (U_maks)"
                ],
                "1. Görsel": [f"{ots1}", f"%{p1*100:.1f}", f"{b1:.3f}", f"{mean_u1:.3f}", f"{max_u1:.3f}"],
                "2. Görsel": [f"{ots2}", f"%{p2*100:.1f}", f"{b2:.3f}", f"{mean_u2:.3f}", f"{max_u2:.3f}"],
                "Net Fark (Δ = Görsel2 - Görsel1)": [
                    f"{ots2-ots1:+d}",
                    f"%{(p2-p1)*100:+.1f}",
                    f"{b2-b1:+.3f}",
                    f"{delta_u:+.3f}",
                    f"{max_u2-max_u1:+.3f}"
                ]
            })

        with tab_cmp_b:
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.metric(
                    label="Ortalama Simetri Kayması (ΔU_ort)",
                    value=f"{delta_u:+.3f}",
                    delta="Simetri Artışı" if delta_u > 0 else ("Asimetri Artışı" if delta_u < 0 else "Değişim Yok")
                )
            with c_col2:
                st.metric(
                    label="Zirve Odak Gücü Değişimi (ΔU_maks)",
                    value=f"{max_u2-max_u1:+.3f}",
                    delta="Odak Güçlendi" if max_u2 > max_u1 else ("Odak Zayıfladı" if max_u2 < max_u1 else "Sabit")
                )

            st.markdown("### 🩺 Klinik Takip & Risk Gelişim Yorumu")
            if abs(delta_u) < 0.08:
                st.info("🟢 **KARARLI SINIR YAPISI ($\\Delta U \\approx 0.00$):** İki muayene arasında anlamlı bir asimetri değişimi kaydedilmemiştir. Sınır yapısı kararlılığını korumaktadır.")
            elif delta_u < -0.08:
                st.error("🔴 **ASİMETRİ ARTIŞI TESPİT EDİLDİ ($\\Delta U < -0.08$):** 2. görselde 1. görsele kıyasla simetri skorunda düşüş (açısal asimetride artış) gerçekleşmiştir. Kötü huylu eğilimli gelişim uyarısı verilir.")
            else:
                st.success("🔵 **DÜZENLİLİK ARTIŞI ($\\Delta U > +0.08$):** 2. görselde simetri oranında artış ve iyileşme eğilimi tespit edilmiştir.")

        with tab_cmp_c:
            st.markdown(r"""
            ### 📖 İkili Karşılaştırma Değerleri Nasıl Yorumlanır?

            #### 1. $\Delta U = U_{ort2} - U_{ort1}$ Fark Skoru Neyi Gösterir?
            * **$\Delta U \approx 0.00$ ($\pm 0.08$ Aralığı):** İki görsel arasında anlamlı bir simetri/asimetri kayması kaydedilmemiştir. Yapı genel biçimsel kararlılığını korur.
            * **$\Delta U < -0.08$ (Negatif Fark / Asimetri Artışı):** 2. görselde 1. görsele kıyasla simetri skorunda belirgin bir düşüş (açısal asimetride artış) gerçekleşmiştir.
            * **$\Delta U > +0.08$ (Pozitif Fark / Düzenlilik Artışı):** 2. görsel 1. görsele kıyasla daha homojen ve yüksek bir dönel simetriye ulaşmıştır.

            #### 2. Isı Haritalarının Kıyaslanması
            * 1. görselde **kırmızı (yüksek simetri)** görünen odağın 2. görselde **maviye/sarıya (düşük simetri)** dönüşmesi, lokal simetri kaybının görsel kanıtıdır.
            """)

# ==============================================================================
# MOD 3: TOPLU GÖRSEL ANALİZİ & İSTATİSTİKSEL HİPOTEZ TESTİ (WELCH'S T-TESTİ)
# ==============================================================================
else:
    st.markdown("---")
    st.subheader("📊 Toplu Görsel Analizi & İstatistiksel Hipotez Testi (Welch's t-Testi)")
    st.caption("Grup A (İyi Huylu / Simetrik) ve Grup B (Kötü Huylu / Asimetrik) numune kümeleri arasında hipotez testi (t-testi) yapın ve veri setini CSV olarak indirin.")

    toplu_mod = st.radio(
        "📌 Toplu Çalışma Tipi Seçin:",
        [
            "🧪 İki Grup İstatistiksel Karşılaştırma (Grup A: İyi Huylu vs Grup B: Kötü Huylu / Şüpheli)",
            "📂 Çoklu Görsel Yükleme & Liste Analizi"
        ]
    )

    k_val_batch = 2 if "2" in k_secim else (5 if "5" in k_secim else (6 if "6" in k_secim else 8))

    if "İki Grup" in toplu_mod:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            n_grup_a = st.number_input("Grup A (İyi Huylu / Simetrik) Numune Sayısı:", 3, 20, 6)
        with col_g2:
            n_grup_b = st.number_input("Grup B (Kötü Huylu / Asimetrik) Numune Sayısı:", 3, 20, 6)

        if st.button("🚀 İki Grup Hipotez Testini Çalıştır", type="primary"):
            progress_bar = st.progress(0)
            toplam_numune = n_grup_a + n_grup_b
            sayac = 0

            grup_a_skorlar = []
            grup_b_skorlar = []
            tum_kayitlar = []

            # Grup A Simülasyonu
            for i in range(n_grup_a):
                res_np = ornek_desen_olustur("Grup A", seed_val=42 + i)
                ots = int(np.mean(res_np)) or 128
                p = float(np.mean(res_np > ots))
                b_val = (p**2) + ((1.0 - p)**2) if b_esik_val is None else b_esik_val
                u_mat, _, _ = hesapla_ysum_matrisi(res_np, k_val_batch, b_val, ots, maske_odakli=roi_filtre)
                non_z = u_mat[u_mat > 0.0]
                m_u = float(np.mean(non_z)) if len(non_z) > 0 else float(np.mean(u_mat))
                grup_a_skorlar.append(m_u)
                tum_kayitlar.append({"Kategori": "Grup A (İyi Huylu)", "Numune": f"A_{i+1}", "YSÜM (U_ort)": round(m_u, 4), "B Eşiği": round(b_val, 4)})
                sayac += 1
                progress_bar.progress(sayac / toplam_numune)

            # Grup B Simülasyonu
            for j in range(n_grup_b):
                res_np = ornek_desen_olustur("Grup B", seed_val=100 + j)
                ots = int(np.mean(res_np)) or 128
                p = float(np.mean(res_np > ots))
                b_val = (p**2) + ((1.0 - p)**2) if b_esik_val is None else b_esik_val
                u_mat, _, _ = hesapla_ysum_matrisi(res_np, k_val_batch, b_val, ots, maske_odakli=roi_filtre)
                non_z = u_mat[u_mat > 0.0]
                m_u = float(np.mean(non_z)) if len(non_z) > 0 else float(np.mean(u_mat))
                grup_b_skorlar.append(m_u)
                tum_kayitlar.append({"Kategori": "Grup B (Kötü Huylu / Şüpheli)", "Numune": f"B_{j+1}", "YSÜM (U_ort)": round(m_u, 4), "B Eşiği": round(b_val, 4)})
                sayac += 1
                progress_bar.progress(sayac / toplam_numune)

            # Welch's t-Testi
            t_stat, p_val = stats.ttest_ind(grup_a_skorlar, grup_b_skorlar, equal_var=False)

            st.markdown("---")
            st.subheader("📈 İki Grup İstatistiksel Karşılaştırma Sonuçları (Welch's t-Testi)")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Grup A Ortalama YSÜM", f"{np.mean(grup_a_skorlar):.4f}")
            res_col2.metric("Grup B Ortalama YSÜM", f"{np.mean(grup_b_skorlar):.4f}")
            res_col3.metric("t-İstatistiği / p-Değeri", f"t={t_stat:.3f} | p={p_val:.4f}")

            if p_val < 0.05:
                st.success(f"🟢 **İSTATİSTİKSEL OLARAK ANLAMLI FARK (p = {p_val:.4f} < 0.05):** YSÜM modeli %95 güven düzeyinde Grup A (İyi Huylu) ve Grup B (Kötü Huylu) lezyon gruplarının asimetri skorlarını birbirinden anlamlı şekilde ayırt etmektedir.")
            else:
                st.warning(f"🟡 **ANLAMLI FARK BULUNAMADI (p = {p_val:.4f} \\ge 0.05):** Gruplar arasında belirgin bir ayrım tespit edilememiştir.")

            # Kutu Grafiği (Boxplot)
            fig_box, ax_box = plt.subplots(figsize=(6, 3.5), dpi=100)
            fig_box.patch.set_facecolor("#0b0f19")
            ax_box.set_facecolor("#0d131f")
            
            bp = ax_box.boxplot([grup_a_skorlar, grup_b_skorlar], labels=["Grup A (İyi Huylu)", "Grup B (Kötü Huylu)"], patch_artist=True)
            colors = ["#00f2fe", "#ff007f"]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set(color="white", linewidth=2)

            ax_box.set_ylabel("YSÜM Skoru (U_ort)", color="white", fontsize=8)
            ax_box.tick_params(colors="white", labelsize=8)
            ax_box.grid(True, alpha=0.2, color="#334155")
            st.pyplot(fig_box)
            plt.close(fig_box)

            # CSV İndirme
            df_grup = pd.DataFrame(tum_kayitlar)
            st.dataframe(df_grup, use_container_width=True)
            csv_data = df_grup.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 İki Grup Test Verilerini CSV Olarak İndir", data=csv_data, file_name="ysum_grup_karshilastirma_raporu.csv", mime="text/csv")

    else:
        uploaded_files = st.file_uploader("📂 Çoklu Görsel Yükleyin:", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        if st.button("🚀 Toplu Analizi Başlat", type="primary"):
            gorsel_listesi = []
            if uploaded_files and len(uploaded_files) > 0:
                for f in uploaded_files:
                    img = Image.open(f).convert("L").resize((200, 200))
                    gorsel_listesi.append((f.name, np.array(img)))
            else:
                test_isimleri = ["Sentetik Düzgün Model", "Sentetik Düzensiz Model", "Sentetik 5-Katlı Yıldız", "Sentetik Rastgele Gürültü"]
                for t in test_isimleri:
                    gorsel_listesi.append((t, ornek_desen_olustur(t)))

            sonuclar = []
            for index, (isim, res_np) in enumerate(gorsel_listesi):
                ots = int(np.mean(res_np)) or 128
                p = float(np.mean(res_np > ots))
                b_val = (p**2) + ((1.0 - p)**2) if b_esik_val is None else b_esik_val
                u_mat, _, _ = hesapla_ysum_matrisi(res_np, k_val_batch, b_val, ots, maske_odakli=roi_filtre)
                non_z = u_mat[u_mat > 0.0]
                m_u = float(np.mean(non_z)) if len(non_z) > 0 else float(np.mean(u_mat))
                max_u = float(np.max(u_mat))

                np.random.seed(42)
                shuffled_np = res_np.copy().flatten()
                np.random.shuffle(shuffled_np)
                u_mat_neg, _, _ = hesapla_ysum_matrisi(shuffled_np.reshape(res_np.shape), k_val_batch, b_val, ots, maske_odakli=False)
                neg_u = float(np.mean(u_mat_neg))

                sonuclar.append({
                    "Görsel Adı / Tanımı": isim,
                    "İkileştirme Eşiği": ots,
                    "Piksel Oranı (p)": round(p, 4),
                    "Şans Eşiği (B)": round(b_val, 4),
                    "Ortalama YSÜM (U_ort)": round(m_u, 4),
                    "Maksimum Odak (U_maks)": round(max_u, 4),
                    "Rastgele Kontrol (U_neg)": round(neg_u, 4),
                    "Göreli Sinyal Farkı (ΔU)": round(m_u - neg_u, 4)
                })

            df_sonuc = pd.DataFrame(sonuclar)
            st.dataframe(df_sonuc, use_container_width=True)
            csv_data = df_sonuc.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Toplu Analiz Raporunu CSV Olarak İndir", data=csv_data, file_name="ysum_toplu_analiz_raporu.csv", mime="text/csv")
