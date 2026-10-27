# ============================================================
# YSÜM (Yerel Simetri Üstünlük Modeli) — Streamlit Web Uygulaması
# Penrose Döşemelerinde Yerel Dönel Simetrinin Şans Eşiğine Göre
# Sayısal Haritalanması — TÜBİTAK 2204-B
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

st.set_page_config(page_title="YSÜM Simetri Analizi", layout="wide")

# ------------------------------------------------------------
# 1) GÖRÜNTÜ İŞLEME FONKSİYONLARI
# ------------------------------------------------------------

def gri_ye_cevir(pil_img, maks_boyut=350):
    """Yüklenen resmi griye çevirir ve performans için küçültür."""
    img = pil_img.convert("L")
    oran = maks_boyut / max(img.size)
    if oran < 1:
        yeni_boyut = (int(img.size[0] * oran), int(img.size[1] * oran))
        img = img.resize(yeni_boyut, Image.LANCZOS)
    return np.array(img, dtype=np.float64)


def otsu_esik_bul(gri_dizi):
    """Otsu yöntemiyle optimum siyah/beyaz ayrım eşiğini otomatik bulur."""
    histogram, _ = np.histogram(gri_dizi, bins=256, range=(0, 256))
    toplam = gri_dizi.size
    toplam_yogunluk = np.dot(np.arange(256), histogram)

    en_iyi_esik = 0
    en_iyi_varyans = 0
    agirlik_arka, toplam_yogunluk_arka = 0, 0

    for esik in range(256):
        agirlik_arka += histogram[esik]
        if agirlik_arka == 0:
            continue
        agirlik_on = toplam - agirlik_arka
        if agirlik_on == 0:
            break
        toplam_yogunluk_arka += esik * histogram[esik]
        ort_arka = toplam_yogunluk_arka / agirlik_arka
        ort_on = (toplam_yogunluk - toplam_yogunluk_arka) / agirlik_on
        varyans_arasi = agirlik_arka * agirlik_on * (ort_arka - ort_on) ** 2
        if varyans_arasi > en_iyi_varyans:
            en_iyi_varyans = varyans_arasi
            en_iyi_esik = esik
    return en_iyi_esik


def tip_haritasi_olustur(gri_dizi):
    """Gri görüntüyü Otsu eşiğine göre ikili (0/1) tip haritasına çevirir."""
    esik = otsu_esik_bul(gri_dizi)
    tip_haritasi = (gri_dizi > esik).astype(np.int8)  # 1 = açık bölge, 0 = koyu bölge
    return tip_haritasi, esik


def icerik_bolgesi_bul(gri_dizi, pencere=9, std_esigi=5.0):
    """
    Görselin düz/tek renk arka plan bölgelerini (desen içermeyen) dışlayan
    bir maske üretir. Yalnızca yerel standart sapması yüksek (desen içeren)
    bölgeler analiz için 'içerik' sayılır.
    """
    yarim = pencere // 2
    h, w = gri_dizi.shape
    maske = np.zeros((h, w), dtype=bool)

    # Basit kayan pencere ile yerel std hesaplama (küçük görseller için yeterli hız)
    kucultme = max(1, pencere // 3)
    for y in range(yarim, h - yarim, kucultme):
        for x in range(yarim, w - yarim, kucultme):
            blok = gri_dizi[y - yarim:y + yarim + 1, x - yarim:x + yarim + 1]
            if blok.std() > std_esigi:
                y0, y1 = max(0, y - kucultme), min(h, y + kucultme + 1)
                x0, x1 = max(0, x - kucultme), min(w, x + kucultme + 1)
                maske[y0:y1, x0:x1] = True
    return maske


# ------------------------------------------------------------
# 2) ŞANS EŞİĞİ (B) HESAPLAMA
# ------------------------------------------------------------

def otomatik_sans_esigi(tip_haritasi, icerik_maskesi=None):
    """
    Her görsele özgü gerçek p ve B değerini hesaplar.
    p = açık bölge oranı (sadece içerik bölgesinde, arka plan hariç)
    B = p^2 + (1-p)^2  -> rastgele eşleşme beklentisi
    """
    if icerik_maskesi is not None and icerik_maskesi.sum() > 0:
        degerler = tip_haritasi[icerik_maskesi]
    else:
        degerler = tip_haritasi.flatten()

    p = degerler.mean()  # açık bölge oranı
    B = p ** 2 + (1 - p) ** 2
    return p, B


# ------------------------------------------------------------
# 3) YEREL DÖNEL SİMETRİ HESAPLAMA (S_k ve U_k)
# ------------------------------------------------------------

def deger_oku(tip_haritasi, y, x):
    """Sınır dışına taşan koordinatları güvenle ele alır."""
    h, w = tip_haritasi.shape
    yi, xi = int(round(y)), int(round(x))
    if 0 <= yi < h and 0 <= xi < w:
        return tip_haritasi[yi, xi]
    return None


def nokta_sk_hesapla(tip_haritasi, cy, cx, k, yaricaplar, aci_sayisi):
    """
    Tek bir merkez nokta (cy, cx) için k-katlı dönel simetri eşleşme oranı S_k.
    Merkez etrafında örnekleme noktaları alınır, 360/k derece döndürülür,
    döndürülmüş noktadaki değer orijinaliyle karşılaştırılır.
    """
    aci_adimi = 2 * np.pi / k
    acilar = np.linspace(0, 2 * np.pi, aci_sayisi, endpoint=False)

    toplam, eslesen = 0, 0
    for r in yaricaplar:
        for theta in acilar:
            y0 = cy + r * np.sin(theta)
            x0 = cx + r * np.cos(theta)
            y1 = cy + r * np.sin(theta + aci_adimi)
            x1 = cx + r * np.cos(theta + aci_adimi)

            v0 = deger_oku(tip_haritasi, y0, x0)
            v1 = deger_oku(tip_haritasi, y1, x1)
            if v0 is None or v1 is None:
                continue
            toplam += 1
            if v0 == v1:
                eslesen += 1

    if toplam == 0:
        return None
    return eslesen / toplam


def hesapla_ysum(tip_haritasi, k, B, icerik_maskesi=None,
                  izgara_adimi=8, yaricap_maks=18, yaricap_adimi=4, aci_sayisi=12):
    """
    Görselin tamamı üzerinde ızgara noktalarında S_k ve U_k hesaplar.
    Dönüş: U_haritasi (2D, NaN=hesaplanamayan/arka plan nokta)
    """
    h, w = tip_haritasi.shape
    yaricaplar = range(4, yaricap_maks, yaricap_adimi)
    kenar_payi = yaricap_maks + 2

    U_haritasi = np.full((h, w), np.nan)

    for cy in range(kenar_payi, h - kenar_payi, izgara_adimi):
        for cx in range(kenar_payi, w - kenar_payi, izgara_adimi):
            if icerik_maskesi is not None and not icerik_maskesi[cy, cx]:
                continue
            Sk = nokta_sk_hesapla(tip_haritasi, cy, cx, k, yaricaplar, aci_sayisi)
            if Sk is None:
                continue
            U = (Sk - B) / (1 - B)
            U_haritasi[cy, cx] = U

    return U_haritasi


# ------------------------------------------------------------
# 4) İSTATİSTİK ÖZETİ VE NEGATİF KONTROL
# ------------------------------------------------------------

def istatistik_ozet(U_haritasi, yuksek_esik=0.5):
    """U haritasından özet istatistikler çıkarır."""
    gecerli = U_haritasi[~np.isnan(U_haritasi)]
    if gecerli.size == 0:
        return {"max": 0, "ortalama": 0, "std": 0, "toplam": 0,
                "yuksek_sayisi": 0, "yuksek_orani": 0}

    yuksek_sayisi = int(np.sum(gecerli > yuksek_esik))
    return {
        "max": float(np.max(gecerli)),
        "ortalama": float(np.mean(gecerli)),
        "std": float(np.std(gecerli)),
        "toplam": int(gecerli.size),
        "yuksek_sayisi": yuksek_sayisi,
        "yuksek_orani": yuksek_sayisi / gecerli.size,
    }


def rastgele_kontrol_uret(tip_haritasi, seed=42):
    """
    Negatif kontrol: aynı açık/koyu oranına sahip ama TAMAMEN RASTGELE
    karıştırılmış bir desen üretir. Gerçek desende simetri 'gerçek' ise,
    bu rastgele desende aynı yüksek U değerleri çıkmamalıdır.
    """
    rng = np.random.default_rng(seed)
    duz = tip_haritasi.flatten().copy()
    rng.shuffle(duz)
    return duz.reshape(tip_haritasi.shape)


# ------------------------------------------------------------
# 5) GÖRSELLEŞTİRME
# ------------------------------------------------------------

def u_haritasi_ciz(U_haritasi, gri_arkaplan, baslik):
    """U haritasını TwoSlopeNorm ile 0 (şans düzeyi) merkezli renklendirir."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(gri_arkaplan, cmap="gray", alpha=0.5)

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)
    im = ax.imshow(U_haritasi, cmap="RdYlBu_r", norm=norm, alpha=0.85)
    ax.set_title(baslik)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="U değeri")
    return fig


# ------------------------------------------------------------
# 6) SONUÇ YORUMLAMA VE RAPORLAMA
# ------------------------------------------------------------

def u_degeri_yorumla(u):
    """Tek bir U değerini sözel olarak yorumlar."""
    if u >= 0.7:
        return "güçlü yerel simetri", "🟢"
    elif u >= 0.4:
        return "orta düzeyde simetri", "🟡"
    elif u >= 0.1:
        return "zayıf / sınırda simetri", "🟠"
    elif u >= -0.1:
        return "şans düzeyinde (simetri iddia edilemez)", "⚪"
    else:
        return "şans düzeyinin altında", "🔴"


def sonuc_raporu_olustur(k_sonuclari, p_degeri, b_degeri, negatif_kontrol_sonuclari=None):
    """Tüm analiz sonuçlarını okunabilir, yorumlanmış bir raporla sunar."""
    st.markdown("---")
    st.header("📋 Sonuç Raporu — Bu Sayılar Ne Anlama Geliyor?")

    en_baskin_k = max(k_sonuclari, key=lambda k: k_sonuclari[k]['max'])
    en_yuksek_u = k_sonuclari[en_baskin_k]['max']
    yorum_metni, yorum_ikon = u_degeri_yorumla(en_yuksek_u)

    st.subheader("1️⃣ Genel Özet")
    st.markdown(f"""
Bu görselde en baskın yerel dönel simetri **{en_baskin_k}-katlı** simetridir.
En yüksek U değeri **{en_yuksek_u:.2f}** çıktı → {yorum_ikon} **{yorum_metni}**.

> U değeri, o noktadaki eşleşme oranının (S_k) **şans eşiğinin (B) ne kadar üzerinde**
> olduğunu, 0 (tam şans düzeyi) ile 1 (kusursuz simetri) arasında ölçeklendirir.
""")

    st.subheader("2️⃣ Tüm Simetri Türleri Karşılaştırması")
    for k, sonuc in sorted(k_sonuclari.items()):
        yorum, ikon = u_degeri_yorumla(sonuc['max'])
        st.markdown(
            f"- **k={k}** → Maks U: {sonuc['max']:.2f} | Ortalama U: {sonuc['ortalama']:.2f} "
            f"(±{sonuc['std']:.2f}) | Yüksek nokta oranı: %{sonuc['yuksek_orani']*100:.1f} | {ikon} {yorum}"
        )

    st.subheader("3️⃣ Bu Sonuç Ne Kadar Güvenilir?")
    en_baskin_sonuc = k_sonuclari[en_baskin_k]
    fark = en_baskin_sonuc['max'] - en_baskin_sonuc['ortalama']
    yuksek_orani = en_baskin_sonuc.get('yuksek_orani', 0)

    if fark > 0.3 and yuksek_orani < 0.05:
        st.warning(f"""
⚠️ **Dikkat:** En yüksek U değeri ({en_baskin_sonuc['max']:.2f}) ile görselin genelindeki
ortalama U değeri ({en_baskin_sonuc['ortalama']:.2f}) arasında büyük fark var; yüksek U'lu
noktalar görselin yalnızca **%{yuksek_orani*100:.1f}**'ine yayılmış.

Bu, genele yayılmış güçlü bir simetriden çok, **tek bir noktada lokal bir eşleşmeden**
kaynaklanıyor olabilir. Bu yüzden yalnızca max değil, ortalama ve yayılım da raporlanmalıdır.
""")
    else:
        st.success(f"""
✅ Yüksek U değerleri görselin **%{yuksek_orani*100:.1f}**'ine yayılmış durumda — bu,
tek bir rastlantısal noktadan değil, görselin genelinde tutarlı bir eğilimden kaynaklandığını
destekler.
""")

    st.subheader("4️⃣ Bu Görsel İçin Şans Eşiği Nasıl Hesaplandı?")
    st.markdown(f"""
Bu görselde açık/koyu bölge oranı **p = {p_degeri:.3f}** olarak ölçüldü (her görsel kendi
oranına sahiptir, sabit bir sayı kullanılmaz).

Şans eşiği: **B = p² + (1-p)² = {b_degeri:.3f}**

> Desende hiçbir simetri olmasa bile, sırf renk/parlaklık dağılımından dolayı rastgele bir
> noktada ortalama **%{b_degeri*100:.0f}** oranında "sahte eşleşme" beklenir. U değeri bu
> sahte eşleşmeyi çıkardıktan sonra kalan gerçek simetri payını gösterir.
""")

    if negatif_kontrol_sonuclari is not None:
        st.subheader("5️⃣ Negatif Kontrol Karşılaştırması")
        kontrol_max = negatif_kontrol_sonuclari[en_baskin_k]['max']
        gercek_max = en_baskin_sonuc['max']
        kontrol_farki = gercek_max - kontrol_max

        st.markdown(f"""
Aynı oranlı ama **tamamen rastgele karıştırılmış** bir kontrol deseni üretildi.
- Gerçek görsel: U = {gercek_max:.2f}
- Rastgele kontrol: U = {kontrol_max:.2f}
- Fark: **{kontrol_farki:.2f}**
""")
        if kontrol_farki > 0.1:
            st.success("✅ Fark anlamlı: gerçek görseldeki simetri, salt rastlantıyla açıklanamayacak kadar yüksek.")
        else:
            st.warning("⚠️ Fark küçük: bu görseldeki yüksek U değeri, rastgele bir desende de benzer şekilde çıkabilir — dikkatli yorumlanmalı.")

    st.subheader("6️⃣ Bu Sonuç Ne DEĞİLDİR?")
    st.info("""
🚫 Bu analiz, yüklediğiniz görselin **Penrose döşemesiyle matematiksel olarak özdeş**
olduğunu ya da **tarihsel olarak Penrose yönteminden türetildiğini** KANITLAMAZ.

Ölçülen şey yalnızca: "bu görseldeki noktalar, şans eşiğine göre beklenenden fazla mı
az mı yerel dönel simetri gösteriyor" sorusudur. Geometrik benzerlik ile tarihsel köken
iddiası farklı şeylerdir; bu site yalnızca birincisini ölçer.
""")


# ------------------------------------------------------------
# 7) STREAMLIT ARAYÜZ
# ------------------------------------------------------------

st.title("🔷 YSÜM — Yerel Simetri Üstünlük Modeli")
st.caption("Penrose Döşemelerinde Yerel Dönel Simetrinin Şans Eşiğine Göre Sayısal Haritalanması")

st.markdown("""
Bu uygulama, yüklediğiniz bir desen/görselde **5, 6 ve 8 katlı yerel dönel simetrinin**
rastgele bir desende beklenenden ne kadar fazla ya da az olduğunu ölçer.
""")

# --- Kenar çubuğu ayarları ---
st.sidebar.header("⚙️ Analiz Ayarları")

b_mod = st.sidebar.radio(
    "Şans Eşiği (B) Modu:",
    ["Otomatik (görselden hesapla) — ÖNERİLEN", "Manuel sabit değer (teorik Penrose: 0.60)"]
)

k_secimi = st.sidebar.multiselect(
    "Analiz edilecek k değerleri (katlı simetri):",
    [3, 4, 5, 6, 7, 8],
    default=[5, 6, 8]
)

izgara_adimi = st.sidebar.slider("Izgara sıklığı (küçük = daha detaylı ama yavaş)", 4, 16, 8)
negatif_kontrol = st.sidebar.checkbox("Negatif kontrol testi ekle (rastgele karşılaştırma)", value=True)

dosya = st.file_uploader("Bir desen/görsel yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if dosya is not None:
    pil_img = Image.open(dosya)
    st.image(pil_img, caption="Yüklenen Görsel", width=300)

    with st.spinner("Görsel işleniyor..."):
        gri_dizi = gri_ye_cevir(pil_img)
        tip_haritasi, otsu_esik = tip_haritasi_olustur(gri_dizi)
        icerik_maskesi = icerik_bolgesi_bul(gri_dizi)

        if b_mod.startswith("Otomatik"):
            p_degeri, b_degeri = otomatik_sans_esigi(tip_haritasi, icerik_maskesi)
        else:
            p_degeri = 0.7236
            b_degeri = 0.60

    st.markdown(f"**Otsu eşiği:** {otsu_esik}  |  **p (açık bölge oranı):** {p_degeri:.3f}  |  **B (şans eşiği):** {b_degeri:.3f}")

    if len(k_secimi) == 0:
        st.warning("Lütfen en az bir k değeri seçin.")
    else:
        k_sonuclari = {}
        negatif_kontrol_sonuclari = {} if negatif_kontrol else None
        kontrol_tip_haritasi = rastgele_kontrol_uret(tip_haritasi) if negatif_kontrol else None

        kolonlar = st.columns(len(k_secimi))

        for i, k in enumerate(k_secimi):
            with st.spinner(f"k={k} için hesaplanıyor..."):
                U_haritasi = hesapla_ysum(tip_haritasi, k, b_degeri, icerik_maskesi,
                                           izgara_adimi=izgara_adimi)
                ozet = istatistik_ozet(U_haritasi)
                k_sonuclari[k] = ozet

                with kolonlar[i]:
                    fig = u_haritasi_ciz(U_haritasi, gri_dizi, f"k={k} Isı Haritası")
                    st.pyplot(fig)
                    st.caption(f"Max U: {ozet['max']:.2f} | Ort: {ozet['ortalama']:.2f}")

                if negatif_kontrol:
                    U_kontrol = hesapla_ysum(kontrol_tip_haritasi, k, b_degeri, None,
                                              izgara_adimi=izgara_adimi)
                    negatif_kontrol_sonuclari[k] = istatistik_ozet(U_kontrol)

        # --- Detaylı sonuç raporu ---
        sonuc_raporu_olustur(k_sonuclari, p_degeri, b_degeri, negatif_kontrol_sonuclari)

else:
    st.info("👆 Analiz için bir görsel yükleyin. Örnek: Penrose döşemesi, Girih deseni, cami çinisi, mozaik fotoğrafı.")

st.markdown("---")
st.caption("TÜBİTAK 2204-B Projesi | Geometrik benzerlik ölçümü — tarihsel köken iddiası içermez.")
