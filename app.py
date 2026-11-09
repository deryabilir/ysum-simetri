import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ==============================================================================
# STREAMLIT SAYFA DÜZENİ
# ==============================================================================
st.set_page_config(
    page_title="YSÜM - Yerel Simetri Üstünlük Modeli",
    page_icon="🌀",
    layout="wide"
)

st.title("🌀 YSÜM - Yerel Simetri Üstünlük Modeli (Revize Akademik Sürüm)")
st.caption("Penrose Geometrisi ve Desen Analizinde Şans Arındırmalı Dönel Simetri Haritalama Laboratuvarı")

# ------------------------------------------------------------------------------
# DETAYLI BİLGİ VE TEORİ PANELİ (BİLİMSEL LİTERATÜR VE DÜZELTİLMİŞ KRONOLOJİ)
# ------------------------------------------------------------------------------
with st.expander("📌 Penrose Geometrisi, Bilimsel Literatür ve Metodolojik Sınırlılıklar", expanded=True):
    tab_teori1, tab_teori2, tab_teori3, tab_teori4, tab_teori5 = st.tabs([
        "📐 Penrose & Kuasikristaller",
        "🧮 YSÜM ($U_k$) ve $B$ Doğrulaması",
        "🌐 Endüstriyel Modelleme",
        "🩺 Tıbbi Asimetri Paralelliği (Etik Sınır)",
        "🕌 Girih Sanatı & Lu-Steinhardt (2007)"
    ])

    with tab_teori1:
        col_t1, col_t2, col_t3 = st.columns()
        with col_t1:
            st.markdown(r"""
            ### 📐 Penrose Döşemeleri ve Kuasikristal Geometrisi
            * **Periyodik Olmayan Yapı (Aperiodic Tiling):** Roger Penrose tarafından 1974'te keşfedilen bu yapılar, düzlemi boşluksuz örter ancak kaydırma (öteleme) simetrisine sahip değildir.
            * **Altın Oran ($\Phi \approx 1.618$):** Karo alan oranları Altın Oran'a yakınsar ($p = 1/\Phi \approx 0.618$).
            * **2011 Nobel Kimya Ödülü İlişkisi:** Dan Shechtman 1982'de kuasikristalleri keşfetmiş ve 2011 Nobel Kimya Ödülü'nü kazanmıştır. Penrose'un matematiksel modeli, doğadaki bu atomik dizilimin teorik çemberini oluşturmuştur.
            """)
        with col_t2:
            fig_n, ax_n = plt.subplots(figsize=(3, 3), dpi=100)
            ax_n.set_facecolor("#0b0f19")
            for ring_r in [0.2, 0.4, 0.65, 0.9]:
                n_dots = 10 if ring_r < 0.8 else 20
                angles_n = np.linspace(0, 2*np.pi, n_dots, endpoint=False)
                ax_n.scatter(ring_r * np.cos(angles_n), ring_r * np.sin(angles_n), c="#00f2fe", s=15, alpha=0.9)
            ax_n.scatter(,, c="white", s=30)
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
        col_m1, col_m2 = st.columns()
        with col_m1:
            st.markdown(r"""
            ### 🧮 Şans Arındırması ve Doğru Cebirsel $B$ Hesabı
            * **İkileştirilmiş Eşleşme ($S_k$):** Aynı ikili matris ($I_{bin}$) üzerinde belirlenen merkez etrafında $2\pi/k$ dönüşlerindeki piksel çakışma oranıdır.
            * **Doğru Penrose Cebirsel Şans Eşiği ($B_{Penrose}$):** Altın oran payı $p = 1/\Phi \approx 0.618$ için:
              $$B = p^2 + (1-p)^2 = (0.618)^2 + (0.382)^2 \approx \mathbf{0.528}$$
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
        ### 🌐 Endüstriyel Modelleme ve Yüzey Analizi
        """)
        col_app1, col_app2, col_app3 = st.columns(3)
        with col_app1:
            st.success("🔬 **1. Malzeme Araştırmaları**\n\nKuasikristal yapılı ince filmlerin yüzey düzenliliğinin ve anizotropisinin deneysel olarak incelenmesi.")
        with col_app2:
            st.info("⚡ **2. Optik & Fotonik Yapılar**\n\nFotonik kristal ızgaralarında açısal simetri sapmalarının simülasyonu.")
        with col_app3:
            st.warning("🏭 **3. Yüzey Kalite Kontrol Modeli**\n\nÜretim yüzeylerinde simetri bozulmalarının sayısal skorlama eşikleriyle bağımsız olarak takip edilmesi.")

    with tab_teori4:
        col_med1, col_med2 = st.columns()
        with col_med1:
            st.markdown(r"""
            ### 🩺 Tıbbi Asimetri Paralelliği ve Metodolojik Etik
            * **Kavramsal Köprü (ABCDE "A" Kriteri):** Dermatolojide lezyon sınırlarının asimetrisi önem taşır. YSÜM modelinin simetriden sapmayı ölçme mantığı bu alanla akademik bir paralellik kurar.
            * **Metodolojik Sınırlılıklar:** Bu yazılım otomatik lezyon segmentasyonu yapmamaktadır; merkez noktaları sabit ızgaradan alınır. Eşik değerleri kavramsal göstergedir.
            * **Kesin Etik Beyan:** Yazılım bir tıbbi teşhis aracı DEĞİLDİR ve klinik testten geçmemiştir; yalnızca matematiksel modelleme sunar.
            """)
        with col_med2:
            fig_m, ax_m = plt.subplots(figsize=(3, 3), dpi=100)
            ax_m.set_facecolor("#f8f9fa")
            theta = np.linspace(0, 2*np.pi, 200)
            r_sym = 0.8 + 0.05 * np.cos(5*theta)
            r_asym = 0.8 + 0.25 * np.sin(3*theta) + 0.1 * np.cos(7*theta)
            ax_m.plot(r_sym * np.cos(theta), r_sym * np.sin(theta), "g--", label="Simetrik Model", lw=1.5)
            ax_m.plot(r_asym * np.cos(theta), r_asym * np.sin(theta), "r-", label="Asimetrik Model", lw=1.8)
            ax_m.set_title("Geometrik Asimetri Şeması", fontsize=8, color="darkred", pad=4)
            ax_m.legend(fontsize=7, loc="lower right")
            ax_m.axis("off")
            st.pyplot(fig_m)
            plt.close(fig_m)

    with tab_teori5:
        col_g1, col_g2 = st.columns()
        with col_g1:
            st.markdown(r"""
            ### 🕌 Girih Sanatı & Lu-Steinhardt (2007) Bulguları
            * **Akademik Literatür:** Peter J. Lu ve Paul J. Steinhardt (2007, *Science*), 15. yüzyıl İslam mimarisindeki (ör. İsfahan Darb-ı İmam) Girih karolarının periyodik olmayan Penrose benzeri geometrik düzenler içerdiğini göstermiştir.
            * **Restorasyon Modeli:** Tarihi yapılarda eksik çini motiflerinin açısal simetri merkezlerinin sayısal olarak haritalanması.
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
# YAN MENÜ (AYARLAR)
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Analiz Ayarları")

analiz_modu = st.sidebar.radio(
    "📌 Çalışma Modu Seçin:",
    ["🖼️ Tekli Görsel Analizi", "🩺 İkili Dermatolojik Karşılaştırma Modu"]
)

k_secim = st.sidebar.selectbox(
    "Simetri Kat Sayısı (k):",
    ["Hepsini Otomatik Tara", "k=2 (İki Eksenli Asimetri)", "k=5 (Penrose/Beşgen)", "k=6 (Altıgen)", "k=8 (Sekizgen)"]
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

negatif_kontrol = st.sidebar.checkbox("Negatif Kontrol (Tekrarlanabilir Piksel Karıştırma)", value=True)

# ------------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR (METODOLOJİK OLARAK DÜZELTİLMİŞ)
# ------------------------------------------------------------------------------
def ornek_desen_olustur(tip):
    np.random.seed(42) # Tekrarlanabilirlik için sabit seed
    w, h = 200, 200
    img_pil = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img_pil)
    cx, cy = w // 2, h // 2
    
    if "Düzgün Sınırlı Model" in tip:
        draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=200, outline=255)
    elif "Düzensiz Sınırlı Model" in tip:
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

# Birebir Aynı İkili Matris Üzerinden Hesaplama Yapan YSÜM Motoru
def hesapla_ysum_matrisi(resim_np, k, b_degeri, ots_thresh, yaricaplar=None, ornek_sayisi=36, adim=8):
    if yaricaplar is None:
        yaricaplar = 
    
    # METODOLOJİK DÜZELTME: Görsel önce ikileştirilir (binarize edilir)
    binary_np = (resim_np > ots_thresh).astype(np.uint8)
    yukseklik, genislik = binary_np.shape
    
    x_grid = np.arange(10, genislik - 10, adim)
    y_grid = np.arange(10, yukseklik - 10, adim)
    u_matrisi = np.zeros((len(y_grid), len(x_grid)))
    donme_acisi = 2 * np.pi / k

    for i, cy in enumerate(y_grid):
        for j, cx in enumerate(x_grid):
            toplam_nokta, eslesen_nokta = 0, 0
            for r in yaricaplar:
                acilar = np.linspace(0, 2 * np.pi, ornek_sayisi, endpoint=False)
                for aci in acilar:
                    toplam_nokta += 1
                    px = int(round(cx + r * np.cos(aci)))
                    py = int(round(cy + r * np.sin(aci)))
                    dpx = int(round(cx + r * np.cos(aci + donme_acisi)))
                    dpy = int(round(cy + r * np.sin(aci + donme_acisi)))
                    
                    if (0 <= px < genislik and 0 <= py < yukseklik and 0 <= dpx < genislik and 0 <= dpy < yukseklik):
                        # Birebir aynı ikili matris karşılaştırması
                        if binary_np[py, px] == binary_np[dpy, dpx]:
                            eslesen_nokta += 1
            if toplam_nokta > 0:
                sk = eslesen_nokta / toplam_nokta
                u_matrisi[i, j] = (sk - b_degeri) / (1.0 - b_degeri) if b_degeri < 1.0 else 0.0
    return u_matrisi, genislik, yukseklik

# ==============================================================================
# MOD 1: TEKLİ GÖRSEL ANALİZİ
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
            "🩺 Sentetik Model 1: Düzgün Sınırlı Model",
            "🩺 Sentetik Model 2: Düzensiz Sınırlı Model",
            "📐 Sentetik Model 3: 5-Katlı Yıldız Desen", 
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
            
            # Aynı ikili matris üzerinden p hesabı
            binary_np = (resim_np > ots_thresh).astype(np.uint8)
            p = float(np.mean(binary_np))
            b_kullanilan = (p**2) + ((1.0 - p)**2) if b_esik_val is None else b_esik_val

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("İkileştirme Eşiği", f"{ots_thresh}")
            m_col2.metric("Piksel Oranı (p)", f"%{p*100:.1f}")
            m_col3.metric("Hesaplanan B Eşiği", f"{b_kullanilan:.3f}")
            st.info(f"💡 Analizde kullanılan B şans eşiği: **{b_kullanilan:.3f}**")

        if st.button("🚀 YSÜM Analizini Başlat", type="primary"):
            st.markdown("---")
            st.subheader("🖼️ Görsel Tabanlı Dönel Simetri Isı Haritası Analizi")

            if "2" in k_secim: k_listesi = 
            elif "5" in k_secim: k_listesi = 
            elif "6" in k_secim: k_listesi = 
            elif "8" in k_secim: k_listesi = 
            else: k_listesi = 

            cols = st.columns(len(k_listesi))
            skorlar, maks_skorlar, top_coords_dict = {}, {}, {}

            for idx, k in enumerate(k_listesi):
                with cols[idx]:
                    u_mat, g, y = hesapla_ysum_matrisi(resim_np, k, b_kullanilan, ots_thresh)
                    max_u = float(np.max(u_mat))
                    mean_u = float(np.mean(u_mat))
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
                    ax.set_title(f"k={k} Katlı Simetri\n(Maks={max_u:.2f}, Ort={mean_u:.2f})", fontsize=10, fontweight="bold")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

            baskin_k = max(skorlar, key=skorlar.get)
            baskin_ort = skorlar[baskin_k]
            baskin_maks = maks_skorlar[baskin_k]
            st.success(f"🏆 **Analiz Bulgusu:** En yüksek ortalama skora sahip yapı **{baskin_k}-Katlı Simetridir** (Ortalama U: **{baskin_ort:.2f}**, Maks: **{baskin_maks:.2f}**)")

            # Rapor Sekmeleri
            st.markdown("---")
            rep_tab1, rep_tab2, rep_tab3, rep_tab4 = st.tabs([
                "📊 Sayısal Özet & Kontrol", "🔍 Odak Koordinatları", "🩺 Kavramsal Dermatolojik Model", "🏭 Yüzey Kalite Modeli"
            ])
            neg_ort = 0.0
            if negatif_kontrol:
                np.random.seed(42) # Tekrarlanabilir seed
                shuffled_test = resim_np.copy().flatten()
                np.random.shuffle(shuffled_test)
                u_mat_test, _, _ = hesapla_ysum_matrisi(shuffled_test.reshape(resim_np.shape), baskin_k, b_kullanilan, ots_thresh)
                neg_ort = float(np.mean(u_mat_test))
            
            fark_u = baskin_ort - neg_ort

            with rep_tab1:
                st.markdown(f"* **Baskın Dönel Simetri:** **{baskin_k}-Katlı**\n* **Ortalama YSÜM Skoru ($U_{{ort}}$):** **{baskin_ort:.3f}** | **Maksimum Odak Skoru ($U_{{maks}}$):** **{baskin_maks:.3f}**\n* **Negatif Kontrol Skoru ($U_{{kontrol}}$):** **{neg_ort:.3f}** (Piksel Shuffling)\n* **Göreli Sinyal Farkı ($\Delta U$):** **+{fark_u:.3f}** (Sinyal/Gürültü Ayrımı)")
            with rep_tab2:
                st.markdown(f"1. **Birincil Odak:** `(X: {top_coords_dict[baskin_k]} px, Y: {top_coords_dict[baskin_k]} px)` — Skor: **{top_coords_dict[baskin_k]:.2f}**\n2. **İkincil Odak:** `(X: {top_coords_dict[baskin_k]} px, Y: {top_coords_dict[baskin_k]} px)` — Skor: **{top_coords_dict[baskin_k]:.2f}**\n3. **Üçüncül Odak:** `(X: {top_coords_dict[baskin_k]} px, Y: {top_coords_dict[baskin_k]} px)` — Skor: **{top_coords_dict[baskin_k]:.2f}**")
            with rep_tab3:
                k2_skor = skorlar.get(2, baskin_ort)
                if k2_skor > 0.45:
                    st.success("🟢 **ŞEMATİK KAVRAMSAL DEĞERLENDİRME: YÜKSEK İKİ EKSENLİ SİMETRİ**\n\nModel sınırlarında belirgin simetri gözlenmektedir. (Not: Bu sonuç bir tıbbi teşhis değildir.)")
                elif k2_skor > 0.15:
                    st.warning("🟡 **ŞEMATİK KAVRAMSAL DEĞERLENDİRME: ORTA DERECE ASİMETRİK SAPMA**\n\nModel sınırlarında kısmi açısal asimetri mevcuttur.")
                else:
                    st.error("🔴 **ŞEMATİK KAVRAMSAL DEĞERLENDİRME: YÜKSEK AÇISAL ASİMETRİ**\n\nModel sınırları simetriden belirgin şekilde sapmaktadır.")
            with rep_tab4:
                if baskin_ort > 0.40: st.success("🟢 **YÜZEY KALİTE MODELİ: BELİRGİN DÖNEL SİMETRİ ($U > 0.40$)**")
                elif baskin_ort > 0.15: st.warning("🟡 **YÜZEY KALİTE MODELİ: KISMİ DÜZEN ($0.15 < U \\le 0.40$)**")
                else: st.error("🔴 **YÜZEY KALİTE MODELİ: DÜZENSİZ / ASİMETRİK YÜZEY ($U \\le 0.15$)**")

# ==============================================================================
# MOD 2: İKİLİ DERMATOLOJİK GÖRSEL KARŞILAŞTIRMA MODU
# ==============================================================================
else:
    st.markdown("---")
    st.subheader("🩺 İkili Dermatolojik Görsel Karşılaştırma Paneli (Kavramsal Model)")
    st.caption("İki ayrı görselin ikili matris seviyesinde YSÜM asimetri değişimini karşılaştırın.")

    col_cmp1, col_cmp2 = st.columns(2)

    with col_cmp1:
        st.markdown("### 🖼️ 1. Görsel (Referans Model)")
        up_file1 = st.file_uploader("1. Resmi Yükleyin", type=["png", "jpg", "jpeg", "webp"], key="up1")
        sel_ornek1 = st.selectbox("Veya 1. Sentetik Modeli Seçin:", [
            "🩺 Sentetik Model 1: Düzgün Sınırlı Model",
            "🩺 Sentetik Model 2: Düzensiz Sınırlı Model",
            "📐 Sentetik Model 3: 5-Katlı Yıldız Desen",
            "🎲 Sentetik Model 5: Rastgele Gürültü"
        ], index=0, key="sel1")

    with col_cmp2:
        st.markdown("### 🖼️ 2. Görsel (Karşılaştırılan Model)")
        up_file2 = st.file_uploader("2. Resmi Yükleyin", type=["png", "jpg", "jpeg", "webp"], key="up2")
        sel_ornek2 = st.selectbox("Veya 2. Sentetik Modeli Seçin:", [
            "🩺 Sentetik Model 2: Düzensiz Sınırlı Model",
            "🩺 Sentetik Model 1: Düzgün Sınırlı Model",
            "📐 Sentetik Model 3: 5-Katlı Yıldız Desen",
            "🎲 Sentetik Model 5: Rastgele Gürültü"
        ], index=0, key="sel2")

    resim1_np, resim2_np = None, None

    if up_file1 is not None:
        img1 = Image.open(up_file1).convert("L")
        img1.thumbnail((220, 220))
        resim1_np = np.array(img1)
    else:
        resim1_np = ornek_desen_olustur(sel_ornek1)

    if up_file2 is not None:
        img2 = Image.open(up_file2).convert("L")
        img2.thumbnail((220, 220))
        resim2_np = np.array(img2)
    else:
        resim2_np = ornek_desen_olustur(sel_ornek2)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.image(resim1_np, caption="1. Görsel Matrisi", width=180, clamp=True)
    with col_v2:
        st.image(resim2_np, caption="2. Görsel Matrisi", width=180, clamp=True)

    if st.button("🚀 İkili Karşılaştırma Analizini Başlat", type="primary"):
        st.markdown("---")
        st.subheader("🔍 Karşılaştırmalı YSÜM Isı Haritası Analizi")

        k_val = 2 if "2" in k_secim else (5 if "5" in k_secim else (6 if "6" in k_secim else 8))

        # 1. Görsel Analiz
        ots1 = int(np.mean(resim1_np))
        if ots1 == 0: ots1 = 128
        bin1 = (resim1_np > ots1).astype(np.uint8)
        p1 = float(np.mean(bin1))
        b1 = (p1**2) + ((1.0 - p1)**2) if b_esik_val is None else b_esik_val
        u_mat1, g1, y1 = hesapla_ysum_matrisi(resim1_np, k_val, b1, ots1)
        mean_u1, max_u1 = float(np.mean(u_mat1)), float(np.max(u_mat1))

        # 2. Görsel Analiz
        ots2 = int(np.mean(resim2_np))
        if ots2 == 0: ots2 = 128
        bin2 = (resim2_np > ots2).astype(np.uint8)
        p2 = float(np.mean(bin2))
        b2 = (p2**2) + ((1.0 - p2)**2) if b_esik_val is None else b_esik_val
        u_mat2, g2, y2 = hesapla_ysum_matrisi(resim2_np, k_val, b2, ots2)
        mean_u2, max_u2 = float(np.mean(u_mat2)), float(np.max(u_mat2))

        out_col1, out_col2 = st.columns(2)

        with out_col1:
            st.markdown("#### 🟢 1. Görsel Isı Haritası")
            fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=100)
            ax1.imshow(resim1_np, cmap="gray", alpha=0.4)
            im1 = ax1.imshow(u_mat1, cmap="RdYlBu_r", extent=[10, g1 - 10, y1 - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("U_k")
            ax1.set_title(f"1. Görsel (k={k_val})\n(Maks={max_u1:.2f}, Ort={mean_u1:.2f})", fontsize=10, fontweight="bold")
            ax1.axis("off")
            st.pyplot(fig1)
            plt.close(fig1)

            st.metric("1. Görsel Ortalama YSÜM", f"{mean_u1:.3f}")
            st.metric("1. Görsel Maksimum Odak", f"{max_u1:.3f}")

        with out_col2:
            st.markdown("#### 🔴 2. Görsel Isı Haritası")
            fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=100)
            ax2.imshow(resim2_np, cmap="gray", alpha=0.4)
            im2 = ax2.imshow(u_mat2, cmap="RdYlBu_r", extent=[10, g2 - 10, y2 - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("U_k")
            ax2.set_title(f"2. Görsel (k={k_val})\n(Maks={max_u2:.2f}, Ort={mean_u2:.2f})", fontsize=10, fontweight="bold")
            ax2.axis("off")
            st.pyplot(fig2)
            plt.close(fig2)

            st.metric("2. Görsel Ortalama YSÜM", f"{mean_u2:.3f}")
            st.metric("2. Görsel Maksimum Odak", f"{max_u2:.3f}")

        # KARŞILAŞTIRMA ÖZET TABLOSU
        st.markdown("---")
        st.subheader("📊 Metrik Karşılaştırma Özeti")

        delta_u = mean_u2 - mean_u1
        st.metric(
            label="Skor Değişim Farkı (ΔU = U_ort2 - U_ort1)",
            value=f"{delta_u:+.3f}",
            delta=f"{delta_u:+.3f} (Pozitif: Simetri Artışı / Negatif: Asimetri Artışı)"
        )

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### 📋 Karşılaştırmalı Metrik Tablosu")
            st.table({
                "Parametre": ["Piksel Oranı (p)", "Hesaplanan Şans Eşiği (B)", "Ortalama YSÜM (U_ort)", "Maksimum Odak (U_maks)"],
                "1. Görsel": [f"%{p1*100:.1f}", f"{b1:.3f}", f"{mean_u1:.3f}", f"{max_u1:.3f}"],
                "2. Görsel": [f"%{p2*100:.1f}", f"{b2:.3f}", f"{mean_u2:.3f}", f"{max_u2:.3f}"],
                "Fark (Δ)": [f"%{(p2-p1)*100:+.1f}", f"{b2-b1:+.3f}", f"{delta_u:+.3f}", f"{max_u2-max_u1:+.3f}"]
            })

        with res_col2:
            st.markdown("### 💡 Göreli Değişim Analizi")
            if abs(delta_u) < 0.08:
                st.info("🟢 **KARARLI GEOMETRİK YAPI:** Görseller arasında belirgin bir simetri/asimetri kayması gözlenmemiştir.")
            elif delta_u < -0.08:
                st.warning("🟡 **ASİMETRİ ARTIŞI TESPİT EDİLDİ:** 2. görselde 1. görsele kıyasla simetri skorunda düşüş (açısal asimetride artış) kaydedilmiştir.")
            else:
                st.success("🔵 **DÜZENLİLİK ARTIŞI:** 2. görselde 1. görsele kıyasla simetri skoru artmıştır.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Bilimsel Not:** Bu laboratuvar yazılımı akademik simülasyon amaçlıdır; tıbbi teşhis veya fabrika otomasyon cihazı değildir.")
