import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

st.set_page_config(page_title="YSÜM Simetri Analizi", layout="wide", page_icon="🌀")

st.title("🌀 YSÜM - Yerel Simetri Üstünlük Modeli")
st.subheader("Görsel Tabanlı Matematiksel Dönel Simetri Analiz Motoru")

with st.expander("📚 Penrose Kaplamaları ve YSÜM Matematiği Hakkında Bilgi", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(r"""
        ### 📐 Penrose Kaplamaları Nedir?
        * **Periyodik Olmayan Yapı:** Düzlemi boşluksuz kaplayan ama kendini asla ötelemeli tekrar etmeyen yapılardır.
        * **Altın Oran ($\Phi \approx 1.618$):** Karo sayı oranları Fibonacci dizisine yakınsar.
        * **5-Katlı Dönel Simetri:** Yapı 72° döndürüldüğünde yerel eşleşme gösterir.
        """)
    with col_b:
        st.markdown(r"""
        ### 🧪 YSÜM ($U_k$) ve Şans Arındırması
        * **Şans Eşiği (B):** Her görsel için **o görselin kendi piksel oranından otomatik hesaplanır** (sabit değildir!).
          Penrose için teorik B≈0,60; farklı bir görselde farklı çıkması normaldir.
        * **Üstünlük Skoru:** $U_k = \frac{S_k - B}{1-B}$
        * **Yorum:** $U_k \approx 1$ mükemmel simetri, $U_k \approx 0$ şans düzeyi, $U_k < 0$ şansın altı.
        """)

st.divider()

# ============== YAN MENÜ ==============
st.sidebar.header("⚙️ Analiz Ayarları")
k_secimi = st.sidebar.selectbox("Simetri Kat Sayısı (k):", [5, 6, 8, "Hepsini Otomatik Tara"], index=3)

b_mod = st.sidebar.radio(
    "Şans Eşiği (B) Modu:",
    ["Otomatik (görselden hesapla) — ÖNERİLEN", "Manuel sabit değer"]
)
if b_mod == "Manuel sabit değer":
    b_esik_manuel = st.sidebar.slider("Manuel B değeri:", 0.0, 1.0, 0.60, 0.05)

negatif_kontrol = st.sidebar.checkbox("🔬 Negatif kontrol ile karşılaştır (rastgele karıştırılmış desen)", value=True)

yuklenen_dosya = st.file_uploader("Analiz edilecek deseni/resmi yükleyin (PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])


# ============== YARDIMCI FONKSİYONLAR ==============

def otsu_esik_bul(gri_dizi):
    """Sabit 128 yerine, görselin kendi histogramına göre en iyi eşiği bulur."""
    histogram, _ = np.histogram(gri_dizi.flatten(), bins=256, range=(0, 256))
    toplam = gri_dizi.size
    toplam_ort = np.dot(np.arange(256), histogram) / toplam
    esik_en_iyi, max_varyans = 0, 0
    agirlik_arka, toplam_arka = 0, 0
    for esik in range(256):
        agirlik_arka += histogram[esik]
        if agirlik_arka == 0:
            continue
        agirlik_on = toplam - agirlik_arka
        if agirlik_on == 0:
            break
        toplam_arka += esik * histogram[esik]
        ort_arka = toplam_arka / agirlik_arka
        ort_on = (toplam_ort * toplam - toplam_arka) / agirlik_on
        varyans = agirlik_arka * agirlik_on * (ort_arka - ort_on) ** 2
        if varyans > max_varyans:
            max_varyans = varyans
            esik_en_iyi = esik
    return esik_en_iyi


def otomatik_sans_esigi(tip_haritasi):
    """B'yi GERÇEK piksel oranından hesaplar (sabit değil)."""
    n1 = np.sum(tip_haritasi == 1)
    toplam = tip_haritasi.size
    p = n1 / toplam
    B = p ** 2 + (1 - p) ** 2
    return p, B


def icerik_bolgesi_bul(tip_haritasi):
    """Arka plan/kenar bölgelerini dışarıda bırakmak için desenin bulunduğu alanı tespit eder."""
    satirlar = np.any(tip_haritasi != tip_haritasi[0, 0], axis=1)
    sutunlar = np.any(tip_haritasi != tip_haritasi[0, 0], axis=0)
    if not satirlar.any():
        return 0, tip_haritasi.shape[0] - 1, 0, tip_haritasi.shape[1] - 1
    y0, y1 = np.where(satirlar)[0][[0, -1]]
    x0, x1 = np.where(sutunlar)[0][[0, -1]]
    return y0, y1, x0, x1


def hesapla_sk(tip_haritasi, cx, cy, k, yaricaplar=(8, 16, 24), ornek_sayisi=36):
    yukseklik, genislik = tip_haritasi.shape
    toplam_nokta, eslesen_nokta = 0, 0
    donme_acisi = 2 * np.pi / k
    for r in yaricaplar:
        acilar = np.linspace(0, 2 * np.pi, ornek_sayisi, endpoint=False)
        for aci in acilar:
            px = int(round(cx + r * np.cos(aci)))
            py = int(round(cy + r * np.sin(aci)))
            dpx = int(round(cx + r * np.cos(aci + donme_acisi)))
            dpy = int(round(cy + r * np.sin(aci + donme_acisi)))
            if (0 <= px < genislik and 0 <= py < yukseklik and
                0 <= dpx < genislik and 0 <= dpy < yukseklik):
                toplam_nokta += 1
                if tip_haritasi[py, px] == tip_haritasi[dpy, dpx]:
                    eslesen_nokta += 1
    return eslesen_nokta / toplam_nokta if toplam_nokta > 0 else np.nan


def harita_hesapla(tip_haritasi, k, B, adim=8):
    yukseklik, genislik = tip_haritasi.shape
    y0, y1, x0, x1 = icerik_bolgesi_bul(tip_haritasi)
    kenar = 26  # en büyük yarıçap + pay

    x_grid = np.arange(max(x0, kenar), min(x1, genislik - kenar), adim)
    y_grid = np.arange(max(y0, kenar), min(y1, yukseklik - kenar), adim)

    U = np.full((len(y_grid), len(x_grid)), np.nan)
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            Sk = hesapla_sk(tip_haritasi, x, y, k)
            if not np.isnan(Sk):
                U[i, j] = (Sk - B) / (1 - B)
    return U, x_grid, y_grid


def istatistik_ozet(U):
    gecerli = U[~np.isnan(U)]
    if gecerli.size == 0:
        return {"maksimum": np.nan, "ortalama": np.nan, "std": np.nan,
                "yuksek_sayisi": 0, "toplam": 0, "yuksek_orani": 0.0}
    return {
        "maksimum": float(np.max(gecerli)),
        "ortalama": float(np.mean(gecerli)),
        "std": float(np.std(gecerli)),
        "yuksek_sayisi": int(np.sum(gecerli > 0.5)),
        "toplam": int(gecerli.size),
        "yuksek_orani": float(np.sum(gecerli > 0.5) / gecerli.size * 100)
    }


def rastgele_kontrol_uret(tip_haritasi, seed=42):
    """Aynı siyah/beyaz oranını koruyan ama konumsal yapıyı yok eden kontrol deseni."""
    rng = np.random.default_rng(seed)
    duz = tip_haritasi.flatten().copy()
    rng.shuffle(duz)
    return duz.reshape(tip_haritasi.shape)


def u_haritasi_ciz(resim_np, U, x_grid, y_grid, baslik):
    yukseklik, genislik = resim_np.shape
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(resim_np, cmap='gray', extent=[0, genislik, yukseklik, 0], alpha=0.4)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)
    im = ax.imshow(U, cmap='RdYlBu_r', norm=norm,
                   extent=[x_grid[0], x_grid[-1], y_grid[-1], y_grid[0]],
                   alpha=0.65, interpolation='bilinear')
    ax.set_title(baslik, fontsize=10, fontweight='bold')
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=0.7, label="U_k")
    return fig


# ============== ANA AKIŞ ==============

if yuklenen_dosya is not None:
    img = Image.open(yuklenen_dosya).convert("L")
    img.thumbnail((200, 200))
    resim_np = np.array(img)

    # Otomatik eşik ile tip haritası (sabit 128 yerine)
    esik = otsu_esik_bul(resim_np)
    tip_haritasi = (resim_np > esik).astype(np.int8)

    p_gercek, B_otomatik = otomatik_sans_esigi(tip_haritasi)

    st.image(yuklenen_dosya, caption="Yüklenen Orijinal Görsel", width=300)

    col_bilgi1, col_bilgi2, col_bilgi3 = st.columns(3)
    col_bilgi1.metric("Otomatik Eşik Değeri", f"{esik}")
    col_bilgi2.metric("Açık Alan Payı (p)", f"{p_gercek:.3f}")
    col_bilgi3.metric("Bu Görsel İçin Gerçek B", f"{B_otomatik:.3f}",
                       help="Penrose teorik beklentisi: p≈0.724, B≈0.60. Bu görsel Penrose değilse farklı çıkması normaldir.")

    B_kullanilacak = B_otomatik if b_mod.startswith("Otomatik") else b_esik_manuel
    if b_mod.startswith("Otomatik"):
        st.success(f"✅ Bu analizde GERÇEK ölçülen B = {B_kullanilacak:.3f} kullanılacak (sabit değer değil).")
    else:
        st.warning(f"⚠️ Manuel B = {B_kullanilacak:.3f} kullanılıyor — bu görselin gerçek B'si {B_otomatik:.3f}'tür, farklıysa yorumlama dikkatli yapılmalı.")

    if st.button("🚀 YSÜM Analizini Başlat", type="primary"):
        k_listesi = [5, 6, 8] if k_secimi == "Hepsini Otomatik Tara" else [k_secimi]

        st.markdown("### 🖼️ Gerçek Görsel Analizi")
        cols = st.columns(len(k_listesi))
        skorlar = {}

        for idx, k in enumerate(k_listesi):
            U, xg, yg = harita_hesapla(tip_haritasi, k, B_kullanilacak)
            ozet = istatistik_ozet(U)
            skorlar[k] = ozet

            fig = u_haritasi_ciz(resim_np, U, xg, yg,
                                  f"k={k} Katlı\nMaks={ozet['maksimum']:.2f} Ort={ozet['ortalama']:.2f}")
            cols[idx].pyplot(fig)
            cols[idx].caption(
                f"Yüksek nokta (U>0.5): {ozet['yuksek_sayisi']}/{ozet['toplam']} "
                f"(%{ozet['yuksek_orani']:.1f})"
            )

        baskin = max(skorlar, key=lambda k: skorlar[k]['ortalama'])
        st.success(
            f"🏆 **Analiz Sonucu:** Ortalama skora göre en baskın simetri **{baskin}-Katlı** "
            f"(Ortalama U: {skorlar[baskin]['ortalama']:.2f}, Maks: {skorlar[baskin]['maksimum']:.2f})"
        )

        # ============== NEGATİF KONTROL ==============
        if negatif_kontrol:
            st.divider()
            st.markdown("### 🔬 Negatif Kontrol: Aynı Oranlı Rastgele Desen")
            st.caption("Aynı siyah/beyaz oranına sahip ama rastgele karıştırılmış bir desen. "
                       "Eğer gerçek görseldeki skor buradakinden belirgin yüksekse, bulduğumuz simetri tesadüf değildir.")

            kontrol_haritasi = rastgele_kontrol_uret(tip_haritasi)
            _, B_kontrol = otomatik_sans_esigi(kontrol_haritasi)

            cols_k = st.columns(len(k_listesi))
            for idx, k in enumerate(k_listesi):
                U_kontrol, xg, yg = harita_hesapla(kontrol_haritasi, k, B_kontrol)
                ozet_kontrol = istatistik_ozet(U_kontrol)

                fig = u_haritasi_ciz(kontrol_haritasi.astype(float) * 255, U_kontrol, xg, yg,
                                      f"KONTROL k={k}\nOrt={ozet_kontrol['ortalama']:.2f}")
                cols_k[idx].pyplot(fig)

                fark = skorlar[k]['ortalama'] - ozet_kontrol['ortalama']
                if abs(fark) > 0.1:
                    cols_k[idx].success(f"✅ Fark: {fark:+.2f} (anlamlı)")
                else:
                    cols_k[idx].warning(f"⚠️ Fark: {fark:+.2f} (zayıf, dikkatli yorumlanmalı)")
else:
    st.info("👆 Lütfen analiz için bir görsel yükleyin.")
