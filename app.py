import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi & lokasi file artefak (semua berada satu folder dengan app.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
FILE_MAPPING = BASE_DIR / "kecamatan_mapping_depok.pkl"
FILE_VARIABEL = BASE_DIR / "variabel_tambahan_depok.csv"
FILE_METADATA = BASE_DIR / "metadata_model_depok.json"

st.set_page_config(page_title="Prediksi Harga Rumah Depok", page_icon="🏠")

LABEL_RENTANG = {
    "Kamar_Tidur": "Kamar Tidur",
    "Kamar_Mandi": "Kamar Mandi",
    "Garasi": "Garasi",
    "Luas_Tanah": "Luas Tanah",
    "Luas_Bangunan": "Luas Bangunan",
}

LABEL_VARIABEL_TAMBAHAN = {
    "jml_faskes": "Jumlah fasilitas kesehatan",
    "jml_stasiun": "Jumlah stasiun",
    "jml_mall": "Jumlah mall",
    "jml_sekolah_pt": "Jumlah sekolah & perguruan tinggi",
    "kepadatan": "Kepadatan penduduk",
}


# ---------------------------------------------------------------------------
# Memuat model & artefak (di-cache: hanya dimuat sekali per server, bukan tiap klik)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model, mohon tunggu...")
def muat_artefak():
    file_model = list(BASE_DIR.glob("model_prediksi_harga_rumah_depok_*.pkl"))
    if len(file_model) != 1:
        raise FileNotFoundError(
            f"Harus ada tepat satu file model 'model_prediksi_harga_rumah_depok_*.pkl' "
            f"di folder aplikasi (ditemukan: {len(file_model)})."
        )

    pipeline = joblib.load(file_model[0])

    # Model dilatih dengan n_jobs=-1. Untuk prediksi satu baris, satu thread lebih ringan
    # dan menghindari pemakaian CPU berlebih di server hosting.
    model_step = getattr(pipeline, "named_steps", {}).get("model")
    if model_step is not None:
        for est in [model_step, *getattr(model_step, "estimators_", [])]:
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1

    kecamatan_mapping = joblib.load(FILE_MAPPING)
    df_variabel = pd.read_csv(FILE_VARIABEL).set_index("kec")
    with open(FILE_METADATA, encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, kecamatan_mapping, df_variabel, metadata


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def format_angka_id(angka, desimal=0):
    '''Format angka gaya Indonesia: titik sebagai pemisah ribuan, koma sebagai desimal.'''
    teks = f"{angka:,.{desimal}f}"
    return teks.replace(",", "_").replace(".", ",").replace("_", ".")


def format_rupiah_ringkas(nilai):
    if nilai >= 1_000_000_000:
        return f"Rp {format_angka_id(nilai / 1_000_000_000, 2)} Miliar"
    return f"Rp {format_angka_id(nilai / 1_000_000, 0)} Juta"


def buat_data_baru(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan,
                   kecamatan, variabel_spasial):
    '''Susun satu baris fitur persis seperti saat training (fitur dasar + fitur turunan + variabel tambahan).'''
    return pd.DataFrame([{
        "Kamar_Tidur": kamar_tidur,
        "Kamar_Mandi": kamar_mandi,
        "Garasi": garasi,
        "Luas_Tanah": luas_tanah,
        "Luas_Bangunan": luas_bangunan,
        "rasio_bangunan_tanah": luas_bangunan / luas_tanah,
        "selisih_luas": luas_tanah - luas_bangunan,
        "total_kamar": kamar_tidur + kamar_mandi,
        "rasio_km_kt": kamar_mandi / kamar_tidur,
        "Kecamatan": kecamatan,
        "jml_faskes": variabel_spasial["jml_faskes"],
        "jml_stasiun": variabel_spasial["jml_stasiun"],
        "jml_mall": variabel_spasial["jml_mall"],
        "jml_sekolah_pt": variabel_spasial["jml_sekolah_pt"],
        "kepadatan": variabel_spasial["kepadatan"],
    }])


# ---------------------------------------------------------------------------
# Tampilan aplikasi
# ---------------------------------------------------------------------------
st.title("🏠 Prediksi Harga Rumah di Depok")
st.write(
    "Masukkan karakteristik rumah, lalu klik **Prediksi Harga**. "
    "Data pendukung wilayah (fasilitas kesehatan, stasiun, mall, sekolah, kepadatan) "
    "terisi otomatis sesuai kecamatan yang dipilih."
)

try:
    pipeline, kecamatan_mapping, df_variabel, metadata = muat_artefak()
except Exception as e:
    st.error(f"Gagal memuat model/artefak: {e}")
    st.stop()

daftar_area = sorted(
    area for area, resmi in kecamatan_mapping.items()
    if resmi in df_variabel.index
)

area = st.selectbox(
    "Kecamatan / Area",
    options=daftar_area,
    index=daftar_area.index("Cinere") if "Cinere" in daftar_area else 0,
    format_func=lambda a: a if kecamatan_mapping[a] == a else f"{a} (Kec. {kecamatan_mapping[a]})",
)
kecamatan_resmi = kecamatan_mapping[area]
variabel_spasial = df_variabel.loc[kecamatan_resmi]

with st.expander(f"Variabel wilayah yang terisi otomatis (Kec. {kecamatan_resmi})"):
    tabel_spasial = pd.DataFrame({
        "Variabel": [LABEL_VARIABEL_TAMBAHAN[k] for k in LABEL_VARIABEL_TAMBAHAN],
        "Nilai": [int(variabel_spasial[k]) for k in LABEL_VARIABEL_TAMBAHAN],
    }).set_index("Variabel")
    st.table(tabel_spasial)

kol1, kol2 = st.columns(2)
with kol1:
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=1, value=100, step=5)
    kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=1, value=3, step=1)
    garasi = st.number_input("Jumlah Garasi", min_value=0, value=1, step=1)
with kol2:
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=1, value=90, step=5)
    kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=1, value=2, step=1)

if st.button("Prediksi Harga", type="primary"):
    # Peringatan bila input berada di luar rentang data latih (model pohon tidak bisa berekstrapolasi)
    nilai_input = {
        "Kamar_Tidur": kamar_tidur, "Kamar_Mandi": kamar_mandi, "Garasi": garasi,
        "Luas_Tanah": luas_tanah, "Luas_Bangunan": luas_bangunan,
    }
    di_luar_rentang = [
        f"{LABEL_RENTANG[kolom]} (data latih: {batas[0]}–{batas[1]})"
        for kolom, batas in metadata["batas_input"].items()
        if not batas[0] <= nilai_input[kolom] <= batas[1]
    ]
    if di_luar_rentang:
        st.warning(
            "Input berikut berada di luar rentang data latih, sehingga estimasi kurang dapat diandalkan: "
            + "; ".join(di_luar_rentang) + "."
        )

    data_baru = buat_data_baru(
        kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, area, variabel_spasial
    )
    prediksi = float(pipeline.predict(data_baru)[0])

    st.metric("Estimasi Harga", format_rupiah_ringkas(prediksi))
    st.caption(f"≈ Rp {format_angka_id(prediksi)}")

metrik = metadata["metrik_uji"]
with st.expander("Tentang model"):
    st.write(
        f"Model: **{metadata['nama_model']}**. Pada data uji, model ini mencapai "
        f"R² {metrik['R2']:.2f} dengan rata-rata kesalahan persentase (MAPE) sekitar "
        f"{metrik['MAPE']:.1f}% dan rata-rata kesalahan absolut (MAE) sekitar "
        f"Rp {format_angka_id(metrik['MAE'] / 1_000_000)} juta."
    )
    st.write(
        "Hasil ini adalah estimasi berdasarkan pola pada data latih, bukan penilaian (appraisal) resmi. "
        "Gunakan sebagai gambaran awal."
    )
