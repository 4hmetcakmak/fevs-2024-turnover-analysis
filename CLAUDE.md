# Proje kuralları (yapay zekâ asistanları ve insanlar için)

Bu repoda kod yazan veya değiştiren herkes (Claude Code dahil) aşağıdaki kurallara uymalıdır.

## Veri sızıntısı
- **Test seti dokunulmazdır.** `06_model.ipynb` içinde en başta `SEED = 42` ile ayrılır, `RandomID`'leri `data/test_ids.csv`'ye yazılır. Hiperparametre, model, eşik veya özellik seçimi için **asla** test seti kullanılmaz.
- Tüm ön işleme (eksik doldurma, ölçekleme, one-hot) **`Pipeline` içinde** olmalı ve yalnızca eğitim verisine `fit` edilmeli. Bölmeden önce yalnızca satır bazlı, istatistik öğrenmeyen dönüşümlere (`hazirla()`) izin verilir.
- Şu sütunlar **özellik olamaz**: `RandomID`, `POSTWT` (yalnızca `sample_weight`), `DLEAVING` (etiket), `level1`, `Q46`, `Q70`, `Q72` (hedefle neredeyse eş anlamlı).
- Yeni bir özellik eklenirse sızıntı taraması (tek başına AUC > 0,75) tekrar çalıştırılmalı.

## Değerlendirme
- Değerlendirme kodu (metrikler, test bölmesi, eşik seçimi) hedefe ulaşmak için **değiştirilmez**.
- Modeller **eşit ayar bütçesiyle** karşılaştırılır (aynı örneklem, aynı CV katları, aynı deneme sayısı).
- Her model en az bir **baseline** ile karşılaştırılır. Baseline'ı anlamlı şekilde geçmeyen model raporlanmaz.
- Tüm metrikler `POSTWT` ile **ağırlıklı** hesaplanır.

## Hataları gizlememe
- Geniş `try/except`, uyarı kapatma (`warnings.filterwarnings("ignore")`) ve açıklamasız `dropna()` yasaktır.
- Satır silen her işlem, öncesi ve sonrası satır sayısını yazdırmalıdır (`log()` fonksiyonu).

## Tekrar üretilebilirlik
- Tüm rastgelelik `SEED = 42` ile sabitlenir.
- Paket sürümleri `requirements.txt` içindedir.
- Model, ön işlemeyle birlikte tek bir `Pipeline` olarak `models/` klasörüne kaydedilir ve geri yükleme testi yapılır.

## Etik
- FEVS anonim bir ankettir. Model ve analizler **bireyleri işaretlemek için değil**, grup düzeyinde içgörü için kullanılır.
- Maskelenmiş demografik değerler rastgele eksik değildir; silinmez, "Eksik" kategorisi olarak tutulur.
