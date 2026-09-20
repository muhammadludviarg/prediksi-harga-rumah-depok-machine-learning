# Prediksi Harga Rumah di Depok

Aplikasi web (Streamlit) untuk mengestimasi harga rumah di Depok berdasarkan luas tanah, luas bangunan,
jumlah kamar tidur, kamar mandi, garasi, dan kecamatan/area. Variabel wilayah pendukung (jumlah fasilitas
kesehatan, stasiun, mall, sekolah & perguruan tinggi, serta kepadatan penduduk) terisi otomatis
berdasarkan kecamatan yang dipilih, sehingga pengguna tidak perlu menginputnya.

Model dilatih pada notebook `dataMining_updated_.ipynb` (kerangka CRISP-DM) dan disimpan sebagai file `.pkl`.

## Struktur repo

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `model_prediksi_harga_rumah_depok_*.pkl` | Pipeline model terlatih (preprocessing + model) |
| `kecamatan_mapping_depok.pkl` | Pemetaan area pemasaran ke kecamatan resmi |
| `variabel_tambahan_depok.csv` | Variabel wilayah per kecamatan resmi |
| `metadata_model_depok.json` | Rentang data latih dan metrik evaluasi model |
| `requirements.txt` | Dependensi (versi sama dengan lingkungan training) |

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Catatan

- File `.pkl` hanya dapat dibuka dengan versi `scikit-learn` yang sama dengan saat training
  (lihat `requirements.txt`). Gunakan juga versi Python yang sama (mayor.minor) saat deployment.
- Hasil prediksi adalah estimasi berdasarkan pola pada data latih, bukan penilaian (appraisal) resmi.
