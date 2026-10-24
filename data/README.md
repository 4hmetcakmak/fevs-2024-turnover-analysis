# Veri nasıl indirilir?

Veri dosyası büyük olduğu için (≈145 MB) bu repoda **yoktur**. GitHub 100 MB'tan büyük dosya kabul etmez; ayrıca verinin sahibi OPM'dir.

## Adımlar

1. OPM'nin FEVS veri sayfasına gidin: <https://www.opm.gov/fevs/public-data-file/>
2. **2024** yılının "Public Release Data File" (PRDF) zip dosyasını indirin (`2024_opm_fevs_prdf_revised.zip`).
3. Zip'i bu klasörün içine açın. Sonuç şöyle görünmeli:

```
data/
└── 2024/
    ├── 2024_OPM_FEVS_PRDF_rv20261007.csv        ← asıl veri (674.207 satır)
    ├── 2024_OPM_FEVS_PRDF_Codebook.xlsx         ← değişkenlerin açıklaması
    ├── 2024_OPM_FEVS_PRDF_Read_Me.docx
    └── 2024_OPM_FEVS_Indices_and_Dimensions_Guide.docx
```

> CSV dosyasının adı OPM'nin yayımladığı revizyona göre değişebilir. Farklıysa `notebooks/01_veri_okuma.ipynb` içindeki `CSV_PATH` satırını güncelleyin.

4. Ek kaynak (isteğe bağlı): **2024 OPM FEVS Technical Report** (ağırlıklandırma, maskeleme ve yanıt oranları burada anlatılır).

## Notebook'lar bu klasöre ne yazar?

| Dosya | Üreten | Açıklama |
|---|---|---|
| `fevs2024_clean.pkl` | 01 | Temizlenmiş, etiketlenmiş veri (diğer tüm notebook'lar bunu okur) |
| `fevs2024_etiketli_skorlu.csv` | 02 | Etiketli tam veri + indeks skorları (Excel'de incelemek için) |
| `test_ids.csv` | 06 | Dondurulmuş test setindeki çalışanların `RandomID` listesi |
