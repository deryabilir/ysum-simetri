# ============================================================
# YSÜM — Yerel Simetri Üstünlük Modeli
# Penrose Döşemelerinde Yerel Dönel Simetrinin Şans Eşiğine Göre
# Sayısal Haritalanması ve Tarihî Geometrik Desenlerle Karşılaştırılması
# TÜBİTAK 2204-B
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

st.set_page_config(page_title="YSÜM — Penrose Simetri Analizi", layout="wide", page_icon="🔷")

PHI = (1 + 5 ** 0.5) / 2

# ============================================================
# BÖLÜM A: ÖRNEK PENROSE DESENİ ÜRETİCİ (vitrin görseli için)
# ============================================================

@st.cache_data
def ornek_penrose_uret(iterasyon=4):
    """
    Robinson üçgenleri (altın gnomon/altın üçgen) ile Penrose P3 deseni üretir.
    Sadece TANITIM/görsel amaçlı, hızlı render için düşük iterasyon kullanılır.
    """
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
            facecolor=renk, edgecolor="white", linewidth=0.2
        )
        ax.add_patch(poligon)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ============================================================
# BÖLÜM B: GÖRÜNTÜ İŞLEME
# ============================================================

def gri_ye_cevir(pil_img, maks_boyut=350):
    img = pil_img.convert("L")
    oran = maks_boyut / max(img.size)
    if oran < 1:
        yeni_boyut = (int(img.size[0] * oran), int(img.size[1] * oran))
        img = img.resize(yeni_boyut, Image.LANCZOS)
    return np.array(img, dtype=np.float64)


def otsu_esik_bul(gri_dizi):
    histogram, _ = np.histogram(gri_dizi, bins=256, range=(0, 256))
    toplam = gri_dizi.size
    toplam_yogunluk = np.dot(np.arange(256), histogram)
    en_iyi_esik, en_iyi_varyans = 0, 0
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
    esik = otsu_esik_bul(gri_dizi)
    return (gri_dizi > esik).astype(np.int8), esik


def icerik_bolgesi_bul(gri_dizi, pencere=9, std_esigi=5.0):
    yarim = pencere // 2
    h, w = gri_dizi.shape
    maske = np.zeros((h, w), dtype=bool)
    kucultme = max(1, pencere // 3)
    for y in range(yarim, h - yarim, kucultme):
        for x in range(yarim, w - yarim, kucultme):
            blok = gri_dizi[y - yarim:y + yarim + 1, x - yarim:x + yarim + 1]
            if blok.std() > std_esigi:
                y0, y1 = max(0, y - kucultme), min(h, y + kucultme + 1)
                x0, x1 = max(0, x - kucultme), min(w, x + kucultme + 1)
                maske[y0:y1, x0:x1] = True
    return maske


# ============================================================
# BÖLÜM C: ŞANS EŞİĞİ (B)
# ============================================================

def otomatik_sans_esigi(tip_haritasi, icerik_maskesi=None):
    if icerik_maskesi is not None and icerik_maskesi.sum() > 0:
        degerler = tip_haritasi[icerik_maskesi]
    else:
        degerler = tip_haritasi.flatten()
    p = degerler.mean()
    B = p ** 2 + (1 - p) ** 2
    return p, B


# ============================================================
# BÖLÜM D: YEREL DÖNEL SİMETRİ (S_k, U_k)
# ============================================================

def deger_oku(tip_haritasi, y, x):
    h, w = tip_haritasi.shape
    yi, xi = int(round(y)), int(round(x))
    if 0 <= yi < h and 0 <= xi < w:
        return tip_haritasi[yi, xi]
    return None


def nokta_sk_hesapla(tip_haritasi, cy, cx, k, yaricaplar, aci_sayisi):
    aci_adimi = 2 * np.pi / k
    acilar = np.linspace(0, 2 * np.pi, aci_sayisi, endpoint=False)
    toplam, eslesen = 0, 0
    for r in yaricaplar:
        for theta in acilar:
            y0 = cy + r * np.sin(theta); x0 = cx + r * np.cos(theta)
            y1 = cy + r * np.sin(theta + aci_adimi); x1 = cx + r * np.cos(theta + aci_adimi)
            v0 = deger_oku(tip_haritasi, y0, x0)
            v1 = deger_oku(tip_haritasi, y1, x1)
            if v0 is None or v1 is None:
                continue
            toplam += 1
            if v0 == v1:
                eslesen += 1
    return eslesen / toplam if toplam else None


def hesapla_ysum(tip_haritasi, k, B, icerik_maskesi=None,
                  izgara_adimi=8, yaricap_maks=18, yaricap_adimi=4, aci_sayisi=12):
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
            U_haritasi[cy, cx] = (Sk - B) / (1 - B)
    return U_haritasi


# ============================================================
# BÖLÜM E: İSTATİSTİK ÖZETİ VE NEGATİF KONTROL
# ============================================================

def istatistik_ozet(U_haritasi, yuksek_esik=0.5):
    gecerli = U_haritasi[~np.isnan(U_haritasi)]
    if gecerli.size == 0:
        return {"max": 0, "ortalama": 0, "std": 0, "toplam": 0, "yuksek_sayisi": 0, "yuksek_orani": 0}
    yuksek_sayisi = int(np.sum(gecerli > yuksek_esik))
    return {
        "max": float(np.max(gecerli)), "ortalama": float(np.mean(gecerli)),
        "std": float(np.std(gecerli)), "toplam": int(gecerli.size),
        "yuksek_sayisi": yuksek_sayisi, "yuksek_orani": yuksek_sayisi / gecerli.size,
    }


def rastgele_kontrol_uret(tip_haritasi, seed=42):
    rng = np.random.default_rng(seed)
    duz = tip_haritasi.flatten().copy()
    rng.shuffle(duz)
    return duz.reshape(tip_haritasi.shape)


# ============================================================
# BÖLÜM F: GÖRSELLEŞTİRME
# ============================================================

def u_haritasi_ciz(U_haritasi, gri_arkaplan, baslik):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(gri_arkaplan, cmap="gray", alpha=0.5)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)
    im = ax.imshow(U_haritasi, cmap="RdYlBu_r", norm=norm, alpha=0.85)
    ax.set_title(baslik)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="U değeri")
    return fig


# ============================================================
# BÖLÜM G: SONUÇ YORUMLAMA
# ============================================================

def u_degeri_yorumla(u):
    if u >= 0.7:   return "güçlü yerel simetri", "🟢"
    elif u >= 0.4: return "orta düzeyde simetri", "🟡"
    elif u >= 0.1: return "zayıf / sınırda simetri", "🟠"
    elif u >= -0.1:return "şans düzeyinde (simetri iddia edilemez)", "⚪"
    else:          return "şans düzeyinin altında", "🔴"


def sonuc_raporu_olustur(k_sonuclari, p_degeri, b_degeri, negatif_kontrol_sonuclari=None):
    st.markdown("---")
    st.header("📋 Sonuç Raporu — Bu Sayılar Ne Anlama Geliyor?")

    en_baskin_k = max(k_sonuclari, key=lambda k: k_sonuclari[k]['max'])
    en_yuksek_u = k_sonuclari[en_baskin_k]['max']
    yorum_metni, yorum_ikon = u_degeri_yorumla(en_yuksek_u)

    st.subheader("1️⃣ Genel Özet")
    st.markdown(f"""
En baskın yerel dönel simetri **{en_baskin_k}-katlı** simetridir. En yüksek U değeri
**{en_yuksek_u:.2f}** → {yorum_ikon} **{yorum_metni}**.

> U, o noktadaki eşleşme oranının (S_k) **şans eşiğinin (B) ne kadar üzerinde** olduğunu,
> 0 (tam şans) ile 1 (kusursuz simetri) arasında ölçeklendirir.
""")

    st.subheader("2️⃣ Tüm Simetri Türleri Karşılaştırması")
    for k, s in sorted(k_sonuclari.items()):
        yorum, ikon = u_degeri_yorumla(s['max'])
        st.markdown(f"- **k={k}** → Maks U: {s['max']:.2f} | Ort: {s['ortalama']:.2f} (±{s['std']:.2f}) "
                    f"| Yüksek nokta oranı: %{s['yuksek_orani']*100:.1f} | {ikon} {yorum}")

    st.subheader("3️⃣ Bu Sonuç Ne Kadar Güvenilir?")
    en_sonuc = k_sonuclari[en_baskin_k]
    fark = en_sonuc['max'] - en_sonuc['ortalama']
    oran = en_sonuc.get('yuksek_orani', 0)
    if fark > 0.3 and oran < 0.05:
        st.warning(f"⚠️ En yüksek U ({en_sonuc['max']:.2f}) ile ortalama ({en_sonuc['ortalama']:.2f}) "
                   f"arasında büyük fark var; yüksek noktalar görselin yalnızca **%{oran*100:.1f}**'inde. "
                   f"Bu, genele yayılmış bir simetriden çok **tek bir lokal eşleşmeden** kaynaklanıyor olabilir.")
    else:
        st.success(f"✅ Yüksek U değerleri görselin **%{oran*100:.1f}**'ine yayılmış — genele yayılmış "
                   f"tutarlı bir eğilimi destekler.")

    st.subheader("4️⃣ Şans Eşiği Bu Görsel İçin Nasıl Hesaplandı?")
    st.markdown(f"""
Açık/koyu bölge oranı **p = {p_degeri:.3f}** olarak ölçüldü (her görsel kendi oranına
sahiptir, sabit sayı kullanılmaz). Şans eşiği: **B = p²+(1-p)² = {b_degeri:.3f}**

> Simetri hiç olmasa bile, sırf renk dağılımından dolayı rastgele bir noktada ortalama
> **%{b_degeri*100:.0f}** oranında "sahte eşleşme" beklenir.
""")

    if negatif_kontrol_sonuclari is not None:
        st.subheader("5️⃣ Negatif Kontrol Karşılaştırması")
        kmax = negatif_kontrol_sonuclari[en_baskin_k]['max']
        gmax = en_sonuc['max']
        st.markdown(f"Gerçek görsel: U = {gmax:.2f} | Rastgele kontrol: U = {kmax:.2f} | Fark: **{gmax-kmax:.2f}**")
        if gmax - kmax > 0.1:
            st.success("✅ Fark anlamlı: gerçek görseldeki simetri rastlantıyla açıklanamaz.")
        else:
            st.warning("⚠️ Fark küçük: bu U değeri rastgele bir desende de benzer çıkabilir.")

    st.subheader("6️⃣ Bu Sonuç Ne DEĞİLDİR?")
    st.info("""
🚫 Bu analiz, görselin **Penrose döşemesiyle matematiksel olarak özdeş** olduğunu ya da
**tarihsel olarak Penrose'tan türediğini** KANITLAMAZ. Yalnızca "bu desen, şans eşiğine göre
beklenenden fazla mı az mı yerel dönel simetri gösteriyor" sorusu ölçülür. Geometrik benzerlik
ile tarihsel köken iddiası farklı şeylerdir.
""")


# ============================================================
# BÖLÜM H: STREAMLIT ARAYÜZ — SEKMELİ YAPI
# ============================================================

st.title("🔷 YSÜM — Yerel Simetri Üstünlük Modeli")
st.caption("Penrose Döşemelerinde Yerel Dönel Simetrinin Şans Eşiğine Göre Sayısal Haritalanması ve Tarihî Geometrik Desenlerle Karşılaştırılması")

sekme_hakkinda, sekme_genel, sekme_detay, sekme_sss = st.tabs(
    ["🏠 Proje Hakkında", "📊 Genel Analiz", "🔬 Detaylı Analiz", "❓ Sıkça Sorulan Sorular"]
)

# ---------------- SEKME 1: PROJE HAKKINDA ----------------
with sekme_hakkinda:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Penrose Döşemesi Nedir?")
        st.markdown("""
Penrose döşemesi, matematikçi **Roger Penrose** tarafından 1974'te tanımlanan,
**periyodik olmayan (aperiyodik)** ama tamamen düzenli bir geometrik desendir.
İki tip rombus karo (kalın 72°-108° ve ince 36°-144°) kullanılarak düzlemi hiçbir
boşluk bırakmadan, ama **asla kendini tekrar etmeden** döşer.

En şaşırtıcı özelliği: desen **5 katlı dönel simetri** içerir — oysa klasik
kristalografi kurallarına göre (Grünbaum & Shephard, 1987) periyodik bir döşemede
5 katlı simetri **imkânsızdır**. Penrose, periyodikliği feda ederek bunu mümkün kıldı.
""")
        st.pyplot(ornek_penrose_uret(iterasyon=4), use_container_width=False)
        st.caption("Bu uygulamanın ürettiği örnek Penrose P3 (rombus) döşemesi")

    with col2:
        st.subheader("Neden Önemli?")
        st.markdown("""
- **1984, Dan Shechtman**: Penrose deseniyle aynı matematiksel yapıya sahip
  *quasicrystal* (kuasikristal) adlı gerçek bir malzeme keşfetti — önceden
  "imkânsız" sayılıyordu. Bu keşif **2011 Nobel Kimya Ödülü**'nü kazandı.
- **2007, Lu & Steinhardt (Science)**: 15. yüzyıl İran'daki Darb-ı İmam Türbesi
  gibi tarihî İslam mimarisinde kullanılan **Girih desenlerinin**, Penrose
  döşemesiyle matematiksel olarak eşdeğer kesme-izdüşüm ilkelerine dayandığını
  gösterdi. *(Not: Bu ilişkiyi ilk keşfeden biz değiliz — bu proje, söz konusu
  bilimsel bulguyu sayısal bir yöntemle yeniden test etmektedir.)*
- Aperiyodik düzen, **sanat, matematik ve fizik arasındaki** nadir
  kesişim noktalarından biridir.
""")

        st.subheader("Günümüzde Nasıl Kullanılıyor?")
        st.markdown("""
- **Malzeme bilimi:** Kuasikristal yapılar, çizilmeye dayanıklı kaplamalar ve
  düşük sürtünmeli yüzeyler (örn. tava kaplamaları, cerrahi aletler) üretiminde kullanılıyor.
- **Mimari ve tasarım:** Zeminde, cephe kaplamalarında ve mozaiklerde "hiç
  tekrar etmeyen ama düzenli" görsel desenler için tercih ediliyor (ör.
  Oxford Matematik Enstitüsü girişi Penrose döşemesiyle kaplıdır).
- **Akustik mühendisliği:** Aperiyodik düzenin ses yansımasını dağıtma
  özelliğinden konser salonu tasarımlarında faydalanılıyor.
- **Eğitim:** Simetri, oran (altın oran φ), sonsuzluk ve sayı dizileri
  (Fibonacci) gibi kavramları somutlaştırmak için matematik eğitiminde kullanılıyor.
""")

    st.markdown("---")
    st.subheader("Bu Proje Ne Yapıyor?")
    st.markdown("""
Bu site, geliştirdiğimiz **YSÜM (Yerel Simetri Üstünlük Modeli)** adlı özgün yöntemle,
yüklediğiniz herhangi bir görseldeki (tarihî desen, çini, mozaik, mimari fotoğraf)
**yerel dönel simetrinin**, rastgele bir desende beklenenden ne kadar fazla ya da az
olduğunu sayısal olarak ölçer.

Yöntemin temelinde **şans eşiği (B)** kavramı var: bir görselde koyu/açık bölge oranı
her zaman belirli bir oranda "sahte" eşleşmeye yol açar — simetri hiç olmasa bile.
Bu modelin hesapladığı **U değeri**, bu şans payını çıkarıp geriye kalan **gerçek**
simetri düzeyini gösterir.
""")

    st.caption("⚠️ Bu araç geometrik benzerlik ölçer; bir desenin Penrose döşemesinden "
               "tarihsel olarak türediğini kanıtlamaz.")

# ---------------- SEKME 2: GENEL ANALİZ (hızlı bakış) ----------------
with sekme_genel:
    st.subheader("📊 Genel Analiz — Hızlı Bakış")
    st.markdown("Bir görsel yükleyin, tek bir k değeri için hızlı sonuç alın.")

    k_genel = st.selectbox("Simetri türü (k):", [5, 6, 8], index=0, key="k_genel")
    dosya_genel = st.file_uploader("Görsel yükleyin", type=["jpg", "jpeg", "png"], key="genel_yukleme")

    if dosya_genel is not None:
        pil_img = Image.open(dosya_genel)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.image(pil_img, caption="Yüklenen Görsel", use_container_width=True)

        with st.spinner("Hesaplanıyor..."):
            gri_dizi = gri_ye_cevir(pil_img)
            tip_haritasi, _ = tip_haritasi_olustur(gri_dizi)
            icerik_maskesi = icerik_bolgesi_bul(gri_dizi)
            p_degeri, b_degeri = otomatik_sans_esigi(tip_haritasi, icerik_maskesi)
            U_haritasi = hesapla_ysum(tip_haritasi, k_genel, b_degeri, icerik_maskesi)
            ozet = istatistik_ozet(U_haritasi)

        with c2:
            fig = u_haritasi_ciz(U_haritasi, gri_dizi, f"k={k_genel} Isı Haritası")
            st.pyplot(fig)

        yorum, ikon = u_degeri_yorumla(ozet["max"])
        st.markdown(f"""
### Sonuç
**En yüksek U değeri: {ozet['max']:.2f}** → {ikon} **{yorum}**

Bu görselde ölçülen açık/koyu oran: p = {p_degeri:.3f} → şans eşiği B = {b_degeri:.3f}

💡 Daha detaylı istatistik, birden fazla k karşılaştırması ve negatif kontrol testi
için **"Detaylı Analiz"** sekmesine geçin.
""")
    else:
        st.info("👆 Hızlı analiz için bir görsel yükleyin.")

# ---------------- SEKME 3: DETAYLI ANALİZ ----------------
with sekme_detay:
    st.subheader("🔬 Detaylı Analiz")
    st.markdown("Birden fazla simetri türünü karşılaştırın, istatistiksel güvenilirliği "
                 "ve negatif kontrol sonucunu görün.")

    st.sidebar.header("⚙️ Detaylı Analiz Ayarları")
    b_mod = st.sidebar.radio(
        "Şans Eşiği (B) Modu:",
        ["Otomatik (görselden hesapla) — ÖNERİLEN", "Manuel sabit değer (teorik Penrose: 0.60)"]
    )
    k_secimi = st.sidebar.multiselect("Analiz edilecek k değerleri:", [3, 4, 5, 6, 7, 8], default=[5, 6, 8])
    izgara_adimi = st.sidebar.slider("Izgara sıklığı (küçük = detaylı ama yavaş)", 4, 16, 8)
    negatif_kontrol = st.sidebar.checkbox("Negatif kontrol testi ekle", value=True)

    dosya_detay = st.file_uploader("Görsel yükleyin", type=["jpg", "jpeg", "png"], key="detay_yukleme")

    if dosya_detay is not None:
        pil_img = Image.open(dosya_detay)
        st.image(pil_img, caption="Yüklenen Görsel", width=300)

        with st.spinner("Görsel işleniyor..."):
            gri_dizi = gri_ye_cevir(pil_img)
            tip_haritasi, 
