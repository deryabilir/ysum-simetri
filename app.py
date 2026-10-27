import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================================================================
# STREAMLIT YSÜM WEB ARAYÜZÜ
# ==============================================================================
st.set_page_config(page_title="YSÜM Simetri Analizi", layout="wide", page_icon="🌀")

st.title("🌀 YSÜM - Yerel Simetri Üstünlük Modeli")
st.subheader("Görsel Tabanlı Matematiksel Dönel Simetri Analiz Motoru")

# Penrose & YSÜM Bilgi Kutusu
with st.expander("📚 Penrose Kaplamaları ve YSÜM Matematiği Hakkında Bilgi", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(r"""
        ### 📐 Penrose Kaplamaları Nedir?
        * **Periyodik Olmayan Yapı:** Penrose kaplamaları, düzlemi hiç boşluk bırakmadan örten ancak kendisini periyodik olarak öteleme simetrisinde tekrarlamayan özel yapılardır.
        * **Altın Oran ($\Phi \approx 1.618$):** Karo sayılarının oranı ve geometrik büyüme adımları doğrudan Altın Oran ve Fibonacci dizisine yakınsar.
        * **5-Katlı Dönel Simetri:** Yapı $72^\circ$ döndürüldüğünde yerel olarak eşleşme gösterir.
        """)
    with col_b:
        st.markdown(r"""
        ### 🧪 YSÜM ($U_k$) ve Şans Arındırması
        * **Şans Eşiği ($B = 0.60$):** Rastgele dağılmış karo/piksel dizilimlerinde bile %60 oranında tesadüfi eşleşme oluşabilir.
        * **Üstünlük Skoru ($U_k$):** $U_k = \frac{S_k - B}{1.0 - B}$ formülü ile rastgele şans hesaptan düşülür.
        * **Skor Yorumu:** $U_k \approx 1.0$ (Mükemmel Simetri), $U_k \le 0.0$ (Şans Seviyesi / Asimetrik).
        """)

st.divider()

# Yan Menü (Ayarlar)
st.sidebar.header("⚙️ Analiz Ayarları")
k_secimi = st.sidebar.selectbox("Simetri Kat Sayısı (k):", [5, 6, 8, "Hepsini Otomatik Tara"], index=3)
b_esik = st.sidebar.slider("Şans Eşiği (B):", 0.0, 1.0, 0.60, 0.05)

# Dosya Yükleme Alanı
yuklenen_dosya = st.file_uploader("Analiz edilecek deseni/resmi yükleyin (PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])

# YSÜM Çekirdek Hesaplama
def hesapla_ysum(resim_np, cx, cy, k, b_esik=0.60, yaricaplar=[8, 16, 24], ornek_sayisi=36):
    yukseklik, genislik = resim_np.shape
    toplam_nokta, eslesen_nokta = 0, 0
    donme_acisi = 2 * np.pi / k

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

    if toplam_nokta == 0: return 0.0
    Sk = eslesen_nokta / toplam_nokta
    return (Sk - b_esik) / (1.0 - b_esik)

if yuklenen_dosya is not None:
    img = Image.open(yuklenen_dosya).convert("L")
    img.thumbnail((200, 200))
    resim_np = np.array(img)
    yukseklik, genislik = resim_np.shape

    st.image(yuklenen_dosya, caption="Yüklenen Orijinal Görsel", width=300)

    if st.button("🚀 YSÜM Analizini Başlat", type="primary"):
        adim = 8
        x_grid = np.arange(10, genislik - 10, adim)
        y_grid = np.arange(10, yukseklik - 10, adim)

        k_listesi = [5, 6, 8] if k_secimi == "Hepsini Otomatik Tara" else [k_secimi]

        cols = st.columns(len(k_listesi))
        skorlar = {}

        for idx, k in enumerate(k_listesi):
            u_matrisi = np.zeros((len(y_grid), len(x_grid)))
            for i, y in enumerate(y_grid):
                for j, x in enumerate(x_grid):
                    u_matrisi[i, j] = hesapla_ysum(resim_np, x, y, k, b_esik)
            
            max_skor = np.max(u_matrisi)
            skorlar[k] = max_skor

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(resim_np, cmap='gray', extent=[0, genislik, yukseklik, 0], alpha=0.5)
            im = ax.imshow(u_matrisi, cmap='jet', extent=[10, genislik - 10, yukseklik - 10, 10], alpha=0.6, vmin=-0.5, vmax=1.0)
            ax.set_title(f"k={k} Katlı (Max U: {max_skor:.2f})")
            ax.axis('off')
            cols[idx].pyplot(fig)

        baskin = max(skorlar, key=skorlar.get)
        st.success(f"🏆 **Analiz Sonucu:** Bu görselde tespit edilen en baskın simetri **{baskin}-Katlı Simetridir** (En yüksek skor: {skorlar[baskin]:.2f}).")
