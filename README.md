# 👥 Employee Attrition Prediction & Analytics Platform
### *PT Jaya Jaya Maju — Data Science Project (DSP)*

Aplikasi analisis dan prediksi kecenderungan *attrition* (pengunduran diri) karyawan berbasis **Machine Learning (Scikit-Learn Random Forest Pipeline)** yang telah dimodifikasi dan siap di-deploy langsung ke **Streamlit Community Cloud**, **Docker**, atau platform cloud lainnya.

---

## 🌟 Fitur Utama Aplikasi

1. **🏢 Profil & Business Understanding**
   - Latar belakang masalah bisnis tingginya turnover karyawan (> 10%) di PT Jaya Jaya Maju.
   - Ringkasan metrik risiko biaya rekrutmen dan strategi retensi preventif.
2. **📊 Dashboard Analitik Interaktif (Looker Studio)**
   - Integrasi embed dashboard Looker Studio langsung di dalam antarmuka Streamlit.
   - Ringkasan insight eksplorasi data (EDA) terkait penyebab utama attrition (lembur, tingkat kompensasi, jarak tempuh rumah, work-life balance).
3. **🔮 Prediksi Attrition Individu**
   - Form input interaktif terstruktur (Demografi, Karir/Pekerjaan, Kepuasan & Work-Life).
   - Tombol **Preset Profil**:
     - 🚨 *Muat Profil Risiko Tinggi (Resign)*
     - 🟢 *Muat Profil Stabil (Bertahan)*
     - 🔄 *Reset ke Nilai Rata-rata*
   - Hasil prediksi instan dengan badge status, probabilitas risiko (risk gauge %), deteksi faktor pemicu spesifik, dan rekomendasi intervensi divisi HR.
4. **📁 Prediksi Batch (Upload CSV / Excel)**
   - Upload file data karyawan secara massal untuk screening risiko.
   - Tersedia tombol download template contoh (`sample_data_attrition.csv`).
   - Ringkasan metrik turnover tim dan tombol export hasil prediksi ke format CSV.
5. **ℹ️ Arsitektur & MLOps Info**
   - Informasi detail spesifikasi model Scikit-Learn Pipeline (`StandardScaler` + `OneHotEncoder` + `RandomForestClassifier`).
   - Integrasi MLOps DagsHub MLflow Artifacts Registry.

---

## 🚀 Panduan Deploy ke Streamlit Community Cloud (Gratis)

Aplikasi ini telah dikonfigurasi penuh dengan dependencies di [`requirements.txt`](file:///d:/CoolYeah/Dekstop/Semester%207/DSP/Project/requirements.txt) dan theme di [`.streamlit/config.toml`](file:///d:/CoolYeah/Dekstop/Semester%207/DSP/Project/.streamlit/config.toml).

### Langkah-langkah Deployment:

1. **Inisialisasi Git & Push ke GitHub**:
   Jika belum di-push ke GitHub, buka terminal di folder project:
   ```bash
   git init
   git add .
   git commit -m "feat: migrate attrition prediction project to Streamlit"
   git branch -M main
   git remote add origin https://github.com/<USERNAME-ANDA>/<NAMA-REPO>.git
   git push -u origin main
   ```
   *(Pastikan file model `model/rf_pipeline.pkl` dan `requirements.txt` ikut ter-push)*.

2. **Buka Streamlit Community Cloud**:
   - Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan login menggunakan akun GitHub Anda.

3. **Buat Aplikasi Baru (New App)**:
   - Klik tombol **"Create app"** atau **"New app"**.
   - Pilih opsi **"Deploy a public app from GitHub"**.
   - Pilih:
     - **Repository**: `<USERNAME-ANDA>/<NAMA-REPO>`
     - **Branch**: `main`
     - **Main file path**: `streamlit_app.py` *(atau `app.py`)*
     - **App URL**: Tentukan URL kustom yang Anda inginkan (opsional).

4. **Deploy**:
   - Klik **"Deploy!"**.
   - Tunggu proses instalasi dependencies selesai (sekitar 1–2 menit).
   - Aplikasi Streamlit Anda sekarang aktif dan dapat diakses publik dari mana saja! 🎉

---

## 💻 Menjalankan Secara Lokal

### 1. Prasyarat
Pastikan Python 3.10+ sudah terpasang.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run streamlit_app.py
```
*atau*:
```bash
streamlit run app.py
```
Buka browser di alamat: `http://localhost:8501`.

---

## 🐳 Menjalankan dengan Docker (Opsional)

Jika ingin menjalankan atau deploy via container Docker (Railway / Render / Cloud Run):

```bash
# Build image
docker build -t attrition-streamlit-app .

# Run container
docker run -p 8501:8501 attrition-streamlit-app
```
Buka `http://localhost:8501`.

---

## 🔄 Menjalankan Versi Legacy Flask (Opsional)

Kode aplikasi Flask terdahulu tetap dipertahankan secara utuh di [`flask_app.py`](file:///d:/CoolYeah/Dekstop/Semester%207/DSP/Project/flask_app.py):

```bash
python flask_app.py
```
Akses di `http://localhost:8000`.

---

## 📂 Struktur Direktori Proyek

```text
├── .streamlit/
│   └── config.toml             # Konfigurasi tema dan server Streamlit
├── model/
│   ├── rf_pipeline.pkl         # Model Scikit-Learn Pipeline Random Forest
│   └── rf_retrain.pkl
├── static/                     # Aset statis CSS / gambar
├── templates/                  # Template HTML Flask terdahulu
├── app.py                      # Unified Streamlit entrypoint
├── streamlit_app.py            # Aplikasi utama Streamlit (Full Features)
├── flask_app.py                # Backup aplikasi Flask terdahulu
├── model_util.py               # Utilitas load model, single pred, batch pred
├── sample_data_attrition.csv   # Contoh template dataset untuk batch upload
├── requirements.txt            # Daftar pustaka Python untuk Streamlit Cloud
├── Dockerfile                  # Konfigurasi container Docker untuk Streamlit
├── Procfile                    # Runner script untuk Railway / Heroku
└── README.md                   # Dokumentasi lengkap proyek
```
