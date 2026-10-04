# 👥 Employee Attrition Prediction & Analytics Platform
### *PT Jaya Jaya Maju — Data Science Project (DSP)*

Aplikasi analisis dan prediksi kecenderungan *attrition* (pengunduran diri) karyawan berbasis **Machine Learning (Scikit-Learn Random Forest Pipeline)** yang siap di-deploy langsung ke **Streamlit Community Cloud** via GitHub.

---

## 🌟 Fitur Utama Aplikasi

1. **🏢 Profil & Business Understanding**
   - Latar belakang masalah turnover karyawan (> 10%) di PT Jaya Jaya Maju.
   - Ringkasan metrik risiko biaya rekrutmen dan strategi retensi preventif.
2. **📊 Dashboard Analitik Interaktif (Looker Studio)**
   - Integrasi embed dashboard Looker Studio langsung di dalam antarmuka Streamlit.
   - Ringkasan insight eksplorasi data (EDA) terkait pemicu utama attrition (lembur rutin, kompensasi rendah, jarak tempuh rumah, work-life balance).
3. **🔮 Prediksi Attrition Individu**
   - Form input terstruktur (Demografi, Karir/Pekerjaan, Kepuasan & Work-Life).
   - Tombol **Preset Profil**:
     - 🚨 *Muat Profil Risiko Tinggi (Resign)*
     - 🟢 *Muat Profil Stabil (Bertahan)*
     - 🔄 *Reset ke Nilai Rata-rata*
   - Hasil prediksi instan dengan badge status, probabilitas risiko (risk gauge %), deteksi faktor pemicu spesifik, dan rekomendasi intervensi divisi HR.
4. **📁 Prediksi Batch (Upload CSV / Excel)**
   - Upload file data karyawan secara massal untuk screening risiko turnover satu departemen.
   - Download template dataset contoh ([`sample_data_attrition.csv`](file:///d:/CoolYeah/Dekstop/Semester%207/DSP/Project/sample_data_attrition.csv)).
   - Ringkasan metrik turnover tim dan tombol download hasil prediksi ke format CSV.
5. **ℹ️ Arsitektur & MLOps Info**
   - Dokumentasi model Scikit-Learn Pipeline (`StandardScaler` + `OneHotEncoder` + `RandomForestClassifier`).
   - Integrasi MLOps DagsHub MLflow Artifacts Registry.

---

## 🚀 Panduan Deploy ke Streamlit Community Cloud (via GitHub)

### 1. Inisialisasi Git & Push ke GitHub
Buka terminal di direktori proyek:
```bash
git add .
git commit -m "chore: clean unused docker and flask files for streamlit cloud"
git branch -M main
git remote add origin https://github.com/<USERNAME-ANDA>/<NAMA-REPO>.git
git push -u origin main
```
*(Pastikan file model `model/rf_pipeline.pkl` dan `requirements.txt` ikut terunggah)*.

### 2. Hubungkan ke Streamlit Community Cloud
1. Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan login menggunakan akun GitHub Anda.
2. Klik tombol **"Create app"** / **"New app"**.
3. Pilih opsi **"Deploy a public app from GitHub"**.
4. Isi form berikut:
   - **Repository**: `<USERNAME-ANDA>/<NAMA-REPO>`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py` *(atau `app.py`)*
5. Klik **"Deploy!"**.
6. Tunggu instalasi dependencies selesai (sekitar 1–2 menit). Web app Streamlit Anda kini live dan dapat diakses publik! 🎉

---

## 💻 Menjalankan Secara Lokal

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan aplikasi Streamlit
streamlit run streamlit_app.py
```
Akses di browser pada: **`http://localhost:8501`**.

---

## 📂 Struktur Direktori Proyek (Clean)

```text
├── .streamlit/
│   └── config.toml             # Konfigurasi tema gelap & server Streamlit
├── model/
│   └── rf_pipeline.pkl         # File model ML (Scikit-Learn Pipeline)
├── .gitignore                  # Mengabaikan virtualenv, pycache, dsb.
├── app.py                      # Entry point alternatif untuk Streamlit Cloud
├── streamlit_app.py            # Aplikasi Streamlit Utama (Full Features)
├── model_util.py               # Modul load model, single pred, & batch pred
├── sample_data_attrition.csv   # Template data karyawan untuk uji coba upload
├── requirements.txt            # Daftar pustaka Python minimal untuk Streamlit
└── README.md                   # Dokumentasi lengkap proyek
```
