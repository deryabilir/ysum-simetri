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

PHI = (1 + 5 ** 0.5) / 2

# ------------------------------------------------------------------------------
# YENİ: ÖRNEK PENROSE GÖRSELİ ÜRETEN FONKSİYON (dosyaya bağımlı değil)
# ------------------------------------------------------------------------------
@st.cache_data
def ornek_penrose_ciz(iterasyon=4):
    """Robinson üçgenleriyle P3 (rombus) tipi Penrose deseni üretir. Sadece tanıtım amaçlı."""
    ucgenler = []
    for i in range(10):
        a1 = i * 2 * np.pi / 10
        a2 = (i + 1) * 2 * np.pi / 10
        A = complex(0, 0)
        B = complex(np.cos(a1), np.sin(a1))
        C = complex(np.cos(a2), np.sin(a2))
        if i % 2 == 0:
            ucgenler.append(("ince", A, C, B))
        else:
            ucgenler.append(("ince", A, B, C))

    def boler(ucgenler):
        yeni = []
        for tip, A, B, C in ucgenler:
            if tip == "ince":
                P = A + (B - A) / PHI
                yeni.append(("ince", C, P, B))
                yeni.append(("kalin", P, C, A))
            else:
                Q = B + (A - B) / PHI
                R = B + (C - B) / PHI
                yeni.append(("kalin", R, C, A))
                yeni.append(("kalin", Q, R, B))
                yeni.append(("ince", R, Q, A))
        return yeni

    for _ in range(iterasyon):
        ucgenler = boler(ucgenler)

    fig, ax = plt.subplots(figsize=(4, 4))
    for tip, A, B, C in ucgenler:
        renk = "#0f205c" if tip == "kalin" else "#008c96"
        poligon = plt.Polygon(
            [(A.real, A.imag), (B.real, B.imag), (C.real, C.imag)],
            facecolor=renk, edgecolor="white", linewidth=0.3
        )
        ax.add_patch(poligon)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig

# ------------------------------------------------------------------------------
# PENROSE VE YSÜM DERS/TEORİ BİLGİ KUTUSU (görsel eklendi)
# ------------------------------------------------------------------------------
with st.expander("📌 Penrose Kaplamaları ve YSÜM Matematiği Hakkında Bilgi", expanded=True):
    col_img, col_a, col_b = st.columns([1, 1.3, 1.3])

    with col_img:
        st.pyplot(ornek_penrose_ciz(iterasyon=4))
        st.caption("Bu uygulamanın kendi ürettiği örnek Penrose P3 (rombus) döşemesi")

    with col_a:
        st.markdown(r"""
        ### 📌 Penrose Kaplamaları Nedir?
        * **Periyodik Olmayan Yapı:** Düzlemi boşluksuz kaplayan ama kendini asla öteleme ile tekrar etmeyen yapılardır.
        * **Altın Oran ($\Phi \approx 1.618$):** Karo sayı oranları Fibonacci dizisine yakınsar.
        * **5-Katlı Dönel Simetri:** Yapı $72^\circ$ döndürüldüğünde yerel eşleşme gösterir.
        
        ### 🔬 Neden Önemlidir ve Günümüzde Nasıl Kullanılır?
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
        st.caption("⚠️ Bu araç geometrik benzerlik ölçer; bir desenin Penrose'tan tarihsel olarak türediğini kanıtlamaz (bkz. Lu & Steinhardt, 2007).")

# ------------------------------------------------------------------------------
# YAN MENÜ (ANALİZ AYARLARI)
# ------------------------------------------------------------------------------
st.sidebar.header("Analiz Ayarları")
k_secim = 
