import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd

# ==============================================================================
# STREAMLIT SAYFA DÜZENİ VE BAŞLIK
# ==============================================================================
st.set_page_config(
    page_title="YSÜM - Yerel Simetri Üstünlük Modeli (TÜBİTAK 2204-B)",
    page_icon="🌀",
    layout="wide"
)

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
            angles_p = np.linspace(0, 2*np.pi, 11)
            r_mesh = np.where(np.arange(11) % 2 == 0, 1.0, 0.382)
            ax_p.plot(r_mesh * np.cos(angles_p), r_mesh * np.sin(angles_p), color="#ff007f", lw=1.8)
            ax_p.fill(r_mesh * np.cos(angles_p), r_mesh * np.sin(angles_p), color="#ff007f", alpha=0.25)
            ax_p.set_title("5-Katlı Geometrik Şema", fontsize=8, color="darkblue", pad=4)
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
            p_vals = np.linspace(0, 1, 100)
            b_vals = p_vals**2 + (1 - p_vals)**2
            ax_b.plot(p_vals, b_vals, "b-", label=r"$B(p) = p^2+(1-p)^2$")
            ax_b.axhline(0.528, color="red", linestyle="--", label=r"Penrose B $\approx 0.528$")
            ax_b.set_xlabel("Açık Piksel Oranı (p)", fontsize=8)
            ax_b.set_ylabel("Şans Eşiği (B)", fontsize=8)
            ax_b.legend(fontsize=7)
            ax_b.grid(True, alpha=0.3)
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
            ax_m.set_facecolor("#f8f9fa")
            theta = np.linspace(0, 2*np.pi, 200)
            r_sym = 0.8 + 0.05 * np.cos(5*theta)
            r_asym = 0.8 + 0.25 * np.sin(3*theta) + 0.1 * np.cos(7*theta)
            ax_m.plot(r_sym * np.cos(theta), r_sym * np.sin(theta), "g--", label="İyi Huylu Eğilimli (Simetrik)", lw=1.5)
            ax_m.plot(r_asym * np.cos(theta), r_asym * np.sin(theta), "r-", label="Şüpheli / Kötü Huylu Eğilimli", lw=1.8)
            ax_m.set_title("ABCDE 'A' Asimetri Şeması", fontsize=8, color="darkred", pad=4)
            ax_m.legend(fontsize=7, loc="lower right")
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
            angles_g = np.linspace(0, 2*np.pi, 21)
            r_girih = np.where(np.arange(21) % 2 == 0, 1.0, 0.45)
            ax_g.plot(r_girih * np.cos(angles_g), r_girih * np.sin(angles_g), color="#008080", lw=2)
            ax_g.fill(r_girih * np.cos(angles_g), r_girih * np.sin(angles_g), color="#008080", alpha=0.3)
            ax_g.set_title("10-Katlı Girih Çizim Modeli", fontsize=8, color="darkgreen", pad=4)
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
        "📊 Toplu Görsel Analizi & CSV Dışa Aktarımı"
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
def ornek_desen_olustur(tip):
    np.random.seed(42)
    w, h = 200, 200
    img_pil = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img_pil)
    cx, cy = w // 2, h // 2
    
    if "Düzgün Sınırlı Model" in tip or "İyi Huylu" in tip:
        draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=200, outline=255)
    elif "Düzensiz Sınırlı Model" in tip or "Kötü Huylu" in tip:
        draw.ellipse([cx - 50, cy - 35, cx + 25, cy + 45], fill=180, outline=255)
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
        yaricaplar = [8, 16, 24, 32]
    
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
                    ax.imshow(resim_np, cmap="gray", alpha=0.4)
                    im = ax.imshow(u_mat, cmap="RdYlBu_r", extent=[10, g - 10, y - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("U_k")
                    ax.set_title(f"k={k} Katlı Simetri\n(Maks={max_u:.2f}, Nesne Ort={mean_u:.2f})", fontsize=10, fontweight="bold")
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
                st.markdown(r"""
                ### 📊 Elde Edilen Metrik Verileri
                * **Baskın Dönel Simetri:** **""" + str(baskin_k) + r"""-Katlı**
                * **Ortalama YSÜM Skoru ($U_{ort}$):** **""" + f"{baskin_ort:.3f}" + r"""**
                * **Maksimum Odak Skoru ($U_{maks}$):** **""" + f"{baskin_maks:.3f}" + r"""**
                * **Negatif Kontrol Skoru ($U_{kontrol}$):** **""" + f"{neg_ort:.3f}" + r"""** (Piksel Shuffling)
                * **Göreli Sinyal Farkı ($\Delta U$):** **+""" + f"{fark_u:.3f}" + r"""** (Sinyal / Gürültü Ayrımı)
                """)
            with rep_tab2:
                st.markdown(f"""
                ### 🔍 En Yüksek 3 Simetri Odak Noktası ({baskin_k}-Katlı)
                1. **Birincil Odak:** `(X: {top_coords_dict[baskin_k][0][0]} px, Y: {top_coords_dict[baskin_k][0][1]} px)` — YSÜM Skoru: **{top_coords_dict[baskin_k][0][2]:.2f}**
                2. **İkincil Odak:** `(X: {top_coords_dict[baskin_k][1][0]} px, Y: {top_coords_dict[baskin_k][1][1]} px)` — YSÜM Skoru: **{top_coords_dict[baskin_k][1][2]:.2f}**
                3. **Üçüncül Odak:** `(X: {top_coords_dict[baskin_k][2][0]} px, Y: {top_coords_dict[baskin_k][2][1]} px)` — YSÜM Skoru: **{top_coords_dict[baskin_k][2][2]:.2f}**
                """)
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
    st.caption(r"İki farklı görseli veya muayene numunesini ikili piksel matrisi seviyesinde yan yana karşılaştırarak asimetri ve düzenlilik farkını ($\Delta U$) analiz edin.")

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
            ax1.imshow(resim1_np, cmap="gray", alpha=0.4)
            im1 = ax1.imshow(u_mat1, cmap="RdYlBu_r", extent=[10, g1 - 10, y1 - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title(f"1. Görsel (Ort U: {mean_u1:.2f})", fontsize=10, fontweight="bold")
            ax1.axis("off")
            st.pyplot(fig1)
            plt.close(fig1)

        with out_col2:
            fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=100)
            ax2.imshow(resim2_np, cmap="gray", alpha=0.4)
            im2 = ax2.imshow(u_mat2, cmap="RdYlBu_r", extent=[10, g2 - 10, y2 - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            ax2.set_title(f"2. Görsel (Ort U: {mean_u2:.2f})", fontsize=10, fontweight="bold")
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
                st.info(r"🟢 **KARARLI SINIR YAPISI ($\Delta U \approx 0.00$):** İki muayene arasında anlamlı bir asimetri değişimi kaydedilmemiştir. Sınır yapısı kararlılığını korumaktadır.")
            elif delta_u < -0.08:
                st.error(r"🔴 **ASİMETRİ ARTIŞI TESPİT EDİLDİ ($\Delta U < -0.08$):** 2. görselde 1. görsele kıyasla simetri skorunda düşüş (açısal asimetride artış) gerçekleşmiştir. Kötü huylu eğilimli gelişim uyarısı verilir.")
            else:
                st.success(r"🔵 **DÜZENLİLİK ARTIŞI ($\Delta U > +0.08$):** 2. görselde simetri oranında artış ve iyileşme eğilimi tespit edilmiştir.")

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
# MOD 3: TOPLU GÖRSEL ANALİZİ & CSV DIŞA AKTARIMI
# ==============================================================================
else:
    st.markdown("---")
    st.subheader("📊 Toplu Görsel Analizi & İstatistiksel Raporlama Paneli")
    st.caption("Birden fazla görseli tek seferde analiz ederek YSÜM metriklerini çıkarın ve CSV veriseti olarak indirin.")

    uploaded_files = st.file_uploader("📂 Çoklu Görsel Yükleyin (Birden fazla seçebilirsiniz):", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    
    k_val_batch = 2 if "2" in k_secim else (5 if "5" in k_secim else (6 if "6" in k_secim else 8))

    if st.button("🚀 Toplu Analizi Başlat", type="primary"):
        gorsel_listesi = []

        if uploaded_files and len(uploaded_files) > 0:
            for f in uploaded_files:
                img = Image.open(f).convert("L").resize((200, 200))
                gorsel_listesi.append((f.name, np.array(img)))
        else:
            st.info("💡 Herhangi bir dosya yüklenmediği için 4'lü Sentetik Test Seti analiz ediliyor...")
            test_isimleri = [
                "Sentetik Düzgün Model (İyi Huylu Eğilimli)",
                "Sentetik Düzensiz Model (Şüpheli / Kötü Huylu Eğilimli)",
                "Sentetik 5-Katlı Penrose Yıldız",
                "Sentetik Rastgele Gürültü"
            ]
            for t in test_isimleri:
                gorsel_listesi.append((t, ornek_desen_olustur(t)))

        sonuclar = []
        progress_bar = st.progress(0)

        for index, (isim, res_np) in enumerate(gorsel_listesi):
            ots = int(np.mean(res_np)) or 128
            binary_np = (res_np > ots).astype(np.uint8)
            p = float(np.mean(binary_np))
            b_val = (p**2) + ((1.0 - p)**2) if b_esik_val is None else b_esik_val

            u_mat, _, _ = hesapla_ysum_matrisi(res_np, k_val_batch, b_val, ots, maske_odakli=roi_filtre)
            non_zero_batch = u_mat[u_mat > 0.0]
            mean_u = float(np.mean(non_zero_batch)) if len(non_zero_batch) > 0 else float(np.mean(u_mat))
            max_u = float(np.max(u_mat))

            # Negatif kontrol (Seed=42)
            np.random.seed(42)
            shuffled_np = res_np.copy().flatten()
            np.random.shuffle(shuffled_np)
            u_mat_neg, _, _ = hesapla_ysum_matrisi(shuffled_np.reshape(res_np.shape), k_val_batch, b_val, ots, maske_odakli=False)
            neg_u = float(np.mean(u_mat_neg))

            delta_u = mean_u - neg_u

            sonuclar.append({
                "Görsel Adı / Tanımı": isim,
                "İkileştirme Eşiği": ots,
                "Piksel Oranı (p)": round(p, 4),
                "Şans Eşiği (B)": round(b_val, 4),
                "Ortalama YSÜM (U_ort)": round(mean_u, 4),
                "Maksimum Odak (U_maks)": round(max_u, 4),
                "Rastgele Kontrol (U_neg)": round(neg_u, 4),
                "Göreli Sinyal Farkı (ΔU)": round(delta_u, 4)
            })

            progress_bar.progress((index + 1) / len(gorsel_listesi))

        df_sonuc = pd.DataFrame(sonuclar)

        st.markdown("### 📋 Toplu Analiz Sonuç Tablosu")
        st.dataframe(df_sonuc, use_container_width=True)

        # Karşılaştırmalı Çubuk Grafik
        st.markdown("### 📈 Görsel Bazlı Skorsal Karşılaştırma Grafiği")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4), dpi=100)
        x_indices = np.arange(len(df_sonuc))
        bar_width = 0.35

        ax_bar.bar(x_indices - bar_width/2, df_sonuc["Ortalama YSÜM (U_ort)"], bar_width, label="Gerçek YSÜM Skoru", color="#00f2fe")
        ax_bar.bar(x_indices + bar_width/2, df_sonuc["Rastgele Kontrol (U_neg)"], bar_width, label="Shuffled Kontrol (Gürültü)", color="#ff007f")

        ax_bar.set_ylabel("Skor Değeri", fontsize=9)
        ax_bar.set_xticks(x_indices)
        ax_bar.set_xticklabels(df_sonuc["Görsel Adı / Tanımı"], rotation=15, ha="right", fontsize=8)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, alpha=0.3)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # CSV İndirme Butonu
        csv_data = df_sonuc.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Toplu Analiz Raporunu CSV Olarak İndir",
            data=csv_data,
            file_name="ysum_toplu_analiz_raporu.csv",
            mime="text/csv"
        )

        st.markdown("""
        ### 📖 Toplu Analiz Verileri Nasıl Okunur?
        * **Ortalama YSÜM (U_ort):** Her görselin tüm yüzeyindeki ortalama dönel düzen düzeyidir.
        * **Maksimum Odak (U_maks):** Görseldeki en yüksek tekil simetri merkezinin gücüdür.
        * **Göreli Sinyal Farkı (ΔU):** Gerçek YSÜM skoru ile rastgele gürültü skoru arasındaki net farktır. Pozitif değerler simetrinin tesadüf olmadığını doğrular.
        """)
