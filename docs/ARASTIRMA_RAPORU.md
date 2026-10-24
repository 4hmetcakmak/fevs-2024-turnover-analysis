# Federal Çalışanlarda Ayrılma Niyetinin Belirleyicileri: 2024 FEVS Verisiyle Makine Öğrenmesi ve Yorumlanabilirlik Analizi

**Yazar:** Ahmet · **Veri:** OPM 2024 Federal Employee Viewpoint Survey (Public Release Data File) · **Tarih:** Ekim 2026

---

## Özet (Abstract)

**Amaç.** Bu çalışma, ABD federal kamu çalışanlarında *bir yıl içinde işten ayrılma niyetini* (DLEAVING) hangi iş deneyimi algılarının en güçlü biçimde öngördüğünü belirlemeyi ve bu niyetin anket cevaplarından ne ölçüde tahmin edilebildiğini ölçmeyi amaçlamaktadır.

**Yöntem.** 2024 FEVS kamuya açık veri dosyasındaki 674.207 yanıt (etiketi dolu 632.298 gözlem; 89 özellik) kullanılmıştır. Tüm tahminler OPM son-tabakalama ağırlıklarıyla (POSTWT) ağırlıklandırılmıştır. Hedefle neredeyse eş anlamlı genel memnuniyet maddeleri (Q46, Q70, Q72) dışlanmış; tek değişkenli AUC > 0,75 kuralıyla otomatik sızıntı taraması yapılmıştır. Dört model (lojistik regresyon, Random Forest, HistGradientBoosting, LightGBM) iki temel modele (sabit tahmin ve tek madde kuralı) karşı, eşit ayar bütçesiyle 3 katlı katmanlı çapraz doğrulamada karşılaştırılmış; nihai değerlendirme yalnızca bir kez, ayrılmış %20'lik test setinde yapılmıştır. Değişken önemi TreeSHAP ve permütasyon önemi ile; kurumlar arası benzerlik hiyerarşik kümeleme ile incelenmiştir.

**Bulgular.** Ağırlıklı ayrılma niyeti oranı %36,8'dir. Seçilen model (HistGradientBoosting) test setinde ROC-AUC = 0,791 [%95 GA 0,788–0,794], PR-AUC = 0,703 ve Brier = 0,174 elde etmiş; tek madde kuralını (AUC 0,695) belirgin biçimde aşmıştır. Ayrılma niyetini en çok açıklayan dört madde iki yorumlama yönteminde de tutarlıdır: **kuruma kişisel bağlılık (Q88), işin ilham vermesi (Q86), ücret memnuniyeti (Q71) ve yeteneklerin kullanılması (Q6)**. Ücret memnuniyeti tek değişkenli korelasyonda 28. sırada iken çok değişkenli modelde ilk üçte yer almakta, yani diğer iş deneyimi algılarından *bağımsız* bir açıklayıcıdır. Demografik farklar istatistiksel olarak anlamlı fakat etki büyüklüğü düşüktür (tüm Cramér's V < 0,08). 40 yaş altı çalışanlarda ücret 1 numaralı etkendir ve federal sektör dışına geçme niyeti olgun kuşağın iki katıdır (%6,5'e karşı %3,1).

**Sonuç.** Ayrılma niyeti, demografiden çok çalışma deneyimi tarafından şekillenmektedir. Anlam, aidiyet ve yeteneklerin kullanımı en güçlü kaldıraçlardır; ücret ise bunlardan bağımsız ve özellikle genç çalışanlarda baskın bir etkendir.

**Anahtar kelimeler:** ayrılma niyeti, çalışan bağlılığı, FEVS, kamu yönetimi, gradient boosting, SHAP, adil model karşılaştırması

---

## 1. Giriş

Kamu kurumlarında personel devri; bilgi kaybı, işe alım maliyeti ve hizmet sürekliliği açısından yüksek maliyetlidir. OPM'nin yıllık FEVS anketi, çalışanlara doğrudan "önümüzdeki yıl içinde ayrılmayı düşünüyor musunuz?" sorusunu yöneltir. Bu çalışma üç araştırma sorusuna yanıt arar:

- **AS1.** Ayrılma niyeti, anket cevaplarından basit kurallara göre anlamlı ölçüde daha iyi tahmin edilebilir mi?
- **AS2.** Diğer tüm maddeler kontrol edildiğinde ayrılma niyetiyle en güçlü ilişkili iş deneyimi algıları hangileridir?
- **AS3.** Bu ilişkiler demografik gruplar, kurum kümeleri ve kuşaklar arasında farklılaşıyor mu?

## 2. Veri

| Özellik | Değer |
|---|---|
| Kaynak | OPM 2024 FEVS Public Release Data File (rv20261007) |
| Yanıtlayan | 674.207 (yanıt oranı ≈ %41) |
| Analize giren (etiketi dolu) | 632.298 (41.909 etiketsiz satır = %6,2 çıkarıldı) |
| Eğitim / test | 505.838 / 126.460 (katmanlı, SEED = 42) |
| Özellik sayısı | 89 (73 Likert madde, Q16 çoklu seçim, Q91, demografi, kurum) |
| Ağırlık | POSTWT (tüm oranlar ve metrikler ağırlıklı) |

Satır sayısı OPM teknik raporundaki tamamlanmış anket sayısıyla birebir eşleşmektedir. "Bilmiyorum" (X) cevapları OPM yöntemine uygun biçimde hesaplamadan çıkarılmış; gizlilik amaçlı maskelenmiş demografik değerler rastgele eksik olmadığı için ayrı bir kategori olarak korunmuştur.

## 3. Yöntem

1. **Sızıntı kontrolü.** Hedefin eş anlamlısı olan Q46/Q70/Q72 dışlandı. Deneysel kontrol: bu maddeler eklendiğinde CV AUC 0,777 → 0,791'e çıkmaktadır; yani model "neden ayrılıyor?" yerine "memnun mu?" sorusunu öğrenirdi. Tek başına AUC > 0,75 olan özellik kalmamıştır (en yüksek 0,696).
2. **Ön işleme.** Eksik doldurma, ölçekleme ve one-hot kodlama tek bir scikit-learn `Pipeline` içinde, yalnızca eğitim verisine fit edilmiştir. Likert maddeler sıralı yapıları gereği sayısal tutulmuştur.
3. **Adil karşılaştırma.** Her model için aynı 60.000 satırlık örneklem, aynı 3 CV katı ve 12 ayar denemesi kullanılmıştır.
4. **Model seçimi.** CV'de lider LightGBM (0,7794 ± 0,0035); HistGradientBoosting (0,7772 ± 0,0033) farkı bir standart sapmanın altında olduğundan, *bir standart hata kuralı* ile daha az bağımlılık gerektiren model seçilmiştir.
5. **Eşik.** F1'i maksimize eden eşik (0,32) eğitim verisindeki out-of-fold tahminlerden belirlenmiştir.
6. **Yorumlama.** LightGBM + TreeSHAP ve lojistik regresyon (ayrı bölmede); nihai model için ağırlıklı permütasyon önemi.
7. **Kümeleme.** 36 kurumun 76 maddelik ağırlıklı % olumlu profili, düzey etkisinden arındırılarak hiyerarşik kümelemeye sokulmuştur.

## 4. Bulgular

### 4.1 Tahmin performansı (AS1)

| Model | CV AUC | Test ROC-AUC [%95 GA] | PR-AUC | Brier |
|---|---|---|---|---|
| Sabit tahmin | 0,500 | 0,500 | 0,369 | 0,233 |
| Tek madde (Q88) | 0,689 | 0,695 [0,691–0,699] | 0,559 | 0,202 |
| Lojistik regresyon | 0,770 | 0,780 [0,777–0,783] | 0,688 | 0,178 |
| Random Forest | 0,769 | 0,783 [0,780–0,786] | 0,690 | 0,178 |
| **HistGradientBoosting** | 0,777 | **0,791 [0,788–0,794]** | 0,703 | **0,174** |
| LightGBM | 0,779 | 0,791 [0,788–0,794] | 0,703 | 0,174 |

Eşik 0,32'de duyarlılık 0,737, kesinlik 0,582, F1 0,650'dir. Kalibrasyon eğrisi köşegene yakındır. Doğrusal model ile en iyi ağaç modeli arasındaki farkın küçük olması (0,011 AUC), ilişkinin büyük ölçüde doğrusal ve toplamsal olduğuna işaret eder.

### 4.2 Ayrılma niyetinin belirleyicileri (AS2)

| Sıra | Madde | SHAP payı | Permütasyon (ΔAUC) | En olumsuz → en olumlu cevapta ayrılma niyeti |
|---|---|---|---|---|
| 1 | Q88 – Kuruma güçlü kişisel bağlılık | %9,8 | 0,016 | %81 → %20 |
| 2 | Q86 – İşim bana ilham veriyor | %7,8 | 0,008 | %82 → %20 |
| 3 | Q71 – Ücret memnuniyeti | %7,8 | 0,016 | %69 → %21 |
| 4 | Q6 – Yeteneklerim iyi kullanılıyor | %7,1 | 0,007 | %77 → %20 |
| 5 | Q47 – Anket sonuçlarının kullanılacağına inanç | %3,9 | 0,001 | %68 → %19 |

Kurum (ΔAUC 0,014) ve federal kıdem (0,009) da önemli yapısal etkenlerdir. Q47'nin permütasyondaki düşük sırası, ilişkili maddelerin önemi paylaşmasından kaynaklanmaktadır; bu nedenle iki yöntemin ortak bulgusu (Q88, Q86, Q71, Q6) en sağlam sonuç olarak değerlendirilmelidir.

### 4.3 Grup farklılıkları (AS3)

- **Demografi:** En büyük fark engellilik durumundadır (%44,4'e karşı %34,5; V = 0,076). Cinsiyet (V = 0,032), ırk (0,043) ve yöneticilik (0,003) etkileri ihmal edilebilir düzeydedir. Demografisi maskelenmiş yanıtlayanlarda niyet %42–45'tir.
- **Kurum kümeleri:** Altı küme bulunmuştur (silhouette ≈ 0,2, orta düzey ayrışma). En yüksek niyet savunma ve güvenlik kümesinde (%41), en düşük finans ve idari hizmet kümesinde (%26).
- **Kritik maddeler:** 75 grubun 37'sinde en düşük puanlı madde Q17 (performans farklarının tanınması); 39'unda ayrılmayla en ilişkili madde Q86'dır.
- **Kuşak:** Genç kuşakta (<40) niyet %38,6, olgun kuşakta %35,4. Gençlerde ücret ve iş-yaşam desteği (Q63) öne çıkarken, olgunlarda kıdem ve emeklilik benzeri "diğer" nedenler (%14,0) belirleyicidir. Model gençlerde daha iyi ayırt etmektedir (AUC 0,817 / 0,778) ve iki grupta da iyi kalibredir.

## 5. Tartışma

Bulgular, iş anlamı ve örgütsel bağlılığın ayrılma niyetinin temel belirleyicileri olduğunu öne süren İş Talepleri–Kaynakları ve örgütsel bağlılık literatürüyle uyumludur. Dikkat çekici bulgu, ücretin "hijyen" etkeni olarak diğer algılara gömülü kalmayıp **bağımsız** bir açıklayıcı olmasıdır: işini anlamlı bulan bir çalışan da ücretten memnun değilse ayrılmayı düşünmektedir. Uygulama açısından, düşük maliyetli kaldıraçlar (takdir, karara katılım, anket sonuçlarına dayalı somut aksiyon) en zayıf puan alan alanlarla örtüşmektedir.

## 6. Sınırlılıklar

1. Etiket *niyettir*, gerçek ayrılma davranışı değildir.
2. Bulgular ilişkiseldir; nedensel yorum için deneysel veya yarı-deneysel tasarım gerekir.
3. Kesitsel tek yıllık veri; zaman içindeki değişim test edilmemiştir.
4. Bölme birey düzeyindedir; modelin daha önce görmediği kurumlara genellenebilirliği (kurum bazlı bölme) sınanmamıştır.
5. Yaş yalnızca iki gruptadır; kuşak analizi kabadır.
6. Ayar araması bütçe nedeniyle 60 bin satırlık örneklemle sınırlıdır.
7. Ortak yöntem varyansı: hem açıklayıcılar hem hedef aynı anketten öz-bildirimdir.

## 7. Sonuç

Ayrılma niyeti anket cevaplarından orta-iyi düzeyde (AUC ≈ 0,79) tahmin edilebilmektedir. En güçlü ve yöntemden bağımsız açıklayıcılar aidiyet, işin anlamı, ücret ve yeteneklerin kullanımıdır; demografinin etkisi küçüktür. Model bireyleri işaretlemek için değil, kurum ve birim düzeyinde iyileştirme önceliklerini belirlemek için kullanılmalıdır.

## Tekrar üretilebilirlik

Tüm kod `notebooks/` altında, sabit sürümler `requirements.txt` içindedir. Ham veri OPM'den indirilmelidir (bkz. `data/README.md`). Ayrıntılı bulgular, Türkiye uygulaması ve KVKK değerlendirmesi ana [README](../README.md) dosyasındadır.
