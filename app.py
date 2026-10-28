import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ==============================================================================
# STREAMLIT SAYFA DÜZENİ
# ==============================================================================
st.set_page_config(
    page_title="YSÜM Simetri Analizi",
    page_icon="🌀",
    layout="wide"
)

st.title("🌀 YSÜM - Yerel Simetri Üstünlük Modeli")
st.caption("Görsel Tabanlı Matematiksel Dönel Simetri Analiz Motoru")

# ------------------------------------------------------------------------------
# PENROSE VE YSÜM DERS/TEORİ BİLGİ KUTUSU (Ekran Görüntülerinizdeki Tasarım)
# ------------------------------------------------------------------------------
with st.expander("📌 Penrose Kaplamaları ve YSÜM Matematiği Hakkında Bilgi", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(r"""
        ### 📌 Penrose Kaplamaları Nedir?
        * **Periyodik Olmayan Yapı:** Düzlemi boşluksuz kaplayan ama kendini asla öteleme ile tekrar etmeyen yapılardır.
        * **Altın Oran ($\Phi \approx 1.618$):** Karo sayı oranları Fibonacci dizisine yakınsar.
        * **5-Katlı Dönel Simetri:** Yapı $72^\circ$ döndürüldüğünde yerel eşleşme gösterir.
        * **Neden Önemlidir ve Günümüzde Nasıl Kullanılır?**
          * **Kuasikristaller (2011 Nobel Kimya Ödülü - Dan Shechtman):** Doğada klasik kristalografi kurallarına uymayan, periyodik olmayan ama uzun menzilli atomik düzenler.
          * **Malzeme Bilimi & Nanoteknoloji:** Yüksek ısı izolasyonu, yapışmaz/sürtünmesiz yüzey kaplamaları, fotonik kristaller ve kuantum kafes yapıları.
          * **Mimari ve Sanat:** 15. yüzyıl Selçuklu ve İslam mimarisinde (İsfahan Darb-ı İmam Türbesi) Girih karoları ile Penrose kaplamalarının matematiksel eşdeğerliği.
        """)
    with col_b:
        st.markdown(r"""
        ### 📐 YSÜM ($U_k$) ve Şans Arındırması
        * **Şans Eşiği ($B$):** Her görsel için o görselin kendi piksel oranından otomatik hesaplanır (sabit değildir). Penrose için teorik $B=0.60$; farklı bir görselde farklı çıkması normaldir.
        * **Üstünlük Skoru:** $U_k = \frac{S_k - B}{1 - B}$
        * **Yorum:** $U_k \approx 1$ mükemmel simetri, $U_k \approx 0$ şans düzeyi, $U_k < 0$ şansın altı.
        """)

# ------------------------------------------------------------------------------
# YAN MENÜ (ANALİZ AYARLARI)
# ------------------------------------------------------------------------------
st.sidebar.header("Analiz Ayarları")
k_secim = st.sidebar.selectbox(
    "Simetri Kat Sayısı (k):",
    ["Hepsini Otomatik Tara", "k=5 (Penrose/Beşgen)", "k=6 (Altıgen)", "k=8 (Sekizgen)"]
)

b_modu = st.sidebar.radio(
    "Şans Eşiği (B) Modu:",
    ["Otomatik (görselden hesapla) — ÖNERİLEN", "Manuel sabit değer"]
)

if "Manuel" in b_modu:
    b_esik_val = st.sidebar.slider("Manuel Şans Eşiği (B):", 0.10, 0.90, 0.60, 0.05)
else:
    b_esik_val = None

negatif_kontrol = st.sidebar.checkbox("Negatif kontrol ile karşılaştır (rastgele karıştırılmış desen)", value=True)

# ------------------------------------------------------------------------------
# GÖRSEL YÜKLEME VE METRİKLER
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Analiz edilecek deseni (resmi) yükleyin (PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])

# YSÜM Çekirdek Hesaplama Fonksiyonu
def hesapla_ysum_matrisi(resim_np, k, b_degeri, yaricaplar=[10, 20, 30, 40], ornek_sayisi=36, adim=8):
    yukseklik, genislik = resim_np.shape
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
                        orj = 1 if resim_np[py, px] > 128 else 0
                        donen = 1 if resim_np[dpy, dpx] > 128 else 0
                        if orj == donen:
                            eslesen_nokta += 1
            if toplam_nokta > 0:
                sk = eslesen_nokta / toplam_nokta
                u_matrisi[i, j] = (sk - b_degeri) / (1.0 - b_degeri) if b_degeri < 1.0 else 0.0
    return u_matrisi, genislik, yukseklik

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img.thumbnail((250, 250))
    resim_np = np.array(img)

    st.image(uploaded_file, caption="Yüklenen Orijinal Görsel", width=180)

    # Otomatik Eşikleme & Piksel Hesabı
    ots_thresh = int(np.mean(resim_np))
    if ots_thresh == 0: ots_thresh = 128
    
    p = float(np.mean(resim_np > ots_thresh))
    if b_esik_val is None:
        b_kullanilan = (p**2) + ((1.0 - p)**2)
    else:
        b_kullanilan = b_esik_val

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Otomatik Eşik Değeri", f"{ots_thresh}")
    m_col2.metric("Açık Alan Payı (p)", f"{p:.3f}")
    m_col3.metric("Bu Görsel İçin Gerçek B", f"{b_kullanilan:.3f}")

    st.success(f"✅ Bu analizde GERÇEK ölçülen B = {b_kullanilan:.3f} kullanılacak (sabit değer değil).")

    if st.button("🚀 YSÜM Analizini Başlat", type="primary"):
        st.markdown("---")
        st.subheader("🖼️ Gerçek Görsel Analizi")

        if "5" in k_secim:
            k_listesi = [5]
        elif "6" in k_secim:
            k_listesi = [6]
        elif "8" in k_secim:
            k_listesi = [8]
        else:
            k_listesi = [5, 6, 8]

        cols = st.columns(len(k_listesi))
        skorlar = {}
        maks_skorlar = {}
        yuksek_noktalar = {}

        for idx, k in enumerate(k_listesi):
            with cols[idx]:
                u_mat, g, y = hesapla_ysum_matrisi(resim_np, k, b_kullanilan)
                max_u = float(np.max(u_mat))
                mean_u = float(np.mean(u_mat))
                skorlar[k] = mean_u
                maks_skorlar[k] = max_u

                toplam_p = u_mat.size
                yuksek_p = int(np.sum(u_mat > 0.5))
                yuzde_p = (yuksek_p / toplam_p) * 100 if toplam_p > 0 else 0
                yuksek_noktalar[k] = (yuksek_p, toplam_p, yuzde_p)

                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                ax.imshow(resim_np, cmap='gray', alpha=0.4)
                im = ax.imshow(u_mat, cmap='RdYlBu_r', extent=[10, g - 10, y - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("U_k")
                ax.set_title(f"k={k} Katlı\nMaks={max_u:.2f} Ort={mean_u:.2f}", fontsize=11, fontweight='bold')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

                st.caption(f"Yüksek nokta (U>0.5): {yuksek_p}/{toplam_p} (%{yuzde_p:.1f})")

        baskin_k = max(skorlar, key=skorlar.get)
        baskin_ort = skorlar[baskin_k]
        baskin_maks = maks_skorlar[baskin_k]

        st.success(f"🏆 **Analiz Sonucu:** Ortalama skora göre en baskın simetri **{baskin_k}-Katlı** (Ortalama U: {baskin_ort:.2f}, Maks: {baskin_maks:.2f})")

        # ----------------------------------------------------------------------
        # NEGATİF KONTROL SEKSİYONU
        # ----------------------------------------------------------------------
        if negatif_kontrol:
            st.markdown("---")
            st.subheader("📉 Negatif Kontrol: Aynı Oranlı Rastgele Desen")
            st.caption("Aynı siyah/beyaz oranına sahip ama rastgele karıştırılmış bir desen. Eğer gerçek görseldeki skor buradakinden belirgin yüksekse, bulduğumuz simetri tesadüf değildir.")

            shuffled_np = resim_np.copy().flatten()
            np.random.shuffle(shuffled_np)
            shuffled_np = shuffled_np.reshape(resim_np.shape)

            cols_neg = st.columns(len(k_listesi))
            for idx, k in enumerate(k_listesi):
                with cols_neg[idx]:
                    u_mat_neg, g, y = hesapla_ysum_matrisi(shuffled_np, k, b_kullanilan)
                    mean_u_neg = float(np.mean(u_mat_neg))

                    fig_neg, ax_neg = plt.subplots(figsize=(4, 4), dpi=100)
                    ax_neg.imshow(shuffled_np, cmap='gray', alpha=0.4)
                    im_neg = ax_neg.imshow(u_mat_neg, cmap='RdYlBu_r', extent=[10, g - 10, y - 10, 10], alpha=0.7, vmin=-0.5, vmax=1.0)
                    cbar_neg = plt.colorbar(im_neg, ax=ax_neg, fraction=0.046, pad=0.04)
                    cbar_neg.set_label("U_k")
                    ax_neg.set_title(f"KONTROL k={k}\nOrt={mean_u_neg:.2f}", fontsize=11, fontweight='bold')
                    ax_neg.axis('off')
                    st.pyplot(fig_neg)
                    plt.close(fig_neg)

                    fark = skorlar[k] - mean_u_neg
                    st.success(f"✅ Fark: +{fark:.2f} (anlamlı)")
