import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Import fungsi utilitas model
from model_util import (
    get_model,
    predict_from_input,
    predict_proba_from_input,
    predict_from_dataframe,
    expected_cols
)

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Employee Attrition Prediction | Jaya Jaya Maju",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan modern, sleek dark theme
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Container */
    .hero-banner {
        background: linear-gradient(135deg, #1b2030 0%, #11141e 100%);
        border: 1px solid rgba(255, 179, 71, 0.25);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    
    .hero-tag {
        display: inline-block;
        background: #ffb347;
        color: #111;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }
    
    .hero-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Card Styles */
    .custom-card {
        background: #181d2a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 179, 71, 0.4);
    }
    
    .card-title {
        color: #ffb347;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    
    /* Result Badges */
    .badge-attrition-yes {
        background: linear-gradient(135deg, #471216 0%, #2b0b0e 100%);
        border: 2px solid #ff4d4d;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 6px 25px rgba(255, 77, 77, 0.2);
    }
    
    .badge-attrition-no {
        background: linear-gradient(135deg, #0e3d23 0%, #072414 100%);
        border: 2px solid #00e676;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 6px 25px rgba(0, 230, 118, 0.2);
    }

    /* Metric pill */
    .metric-pill {
        display: inline-block;
        background: rgba(255, 179, 71, 0.15);
        border: 1px solid rgba(255, 179, 71, 0.4);
        color: #ffb347;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL LOADER DENGAN CACHE STREAMLIT
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model Machine Learning...")
def load_cached_model():
    """Memuat model Random Forest Pipeline (Lokal atau DagsHub)."""
    try:
        model = get_model()
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_cached_model()

# ==============================================================================
# 3. SIDEBAR NAVIGATION & STATUS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
        <h2 style="color: #ffb347; margin:0; font-size: 1.5rem;">🏢 Jaya Jaya Maju</h2>
        <p style="color: #a0aec0; font-size: 0.85rem; margin: 0.2rem 0 0 0;">HR Analytics & Retention AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "📌 Menu Navigasi",
        [
            "🏢 Profil & Business Understanding",
            "📊 Dashboard Analitik (Looker)",
            "🔮 Prediksi Attrition (Individu)",
            "📁 Prediksi Batch (Upload CSV)",
            "ℹ️ Info Model & Arsitektur"
        ],
        index=2  # Default ke prediksi individu
    )
    
    st.markdown("---")
    
    # Status Model Card di Sidebar
    st.markdown("##### ⚙️ Status Sistem")
    if model is not None:
        st.success("✅ Model: **Aktif (Random Forest)**")
        st.caption("Pipeline: StandardScaler + OneHotEncoder + RandomForest")
    else:
        st.error(f"❌ Model Gagal Dimuat: {load_error}")
        if st.button("🔄 Muat Ulang Model"):
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.caption("Dikembangkan untuk Final Project Data Science / DSP • Streamlit Cloud Ready 🚀")


# ==============================================================================
# 4. HALAMAN 1: PROFIL & BUSINESS UNDERSTANDING
# ==============================================================================
if menu == "🏢 Profil & Business Understanding":
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-tag">Business Understanding</span>
        <h1 class="hero-title">Employee Attrition Analysis</h1>
        <p class="hero-subtitle">
            Memahami faktor-faktor penentu pengunduran diri karyawan dan merancang strategi retensi berbasis Machine Learning pada PT Jaya Jaya Maju.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Tingkat Attrition Baseline", value="> 10%", delta="Di atas batas aman", delta_color="inverse")
    with col2:
        st.metric(label="Target Retensi Karyawan", value="≥ 88%", delta="+3.5% vs Q3")
    with col3:
        st.metric(label="Biaya Penggantian", value="6 - 9x", help="Rata-rata 6-9 bulan gaji karyawan per turnover")
    with col4:
        st.metric(label="Akurasi Pipeline Model", value="~87%", delta="Random Forest")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("""
        <div class="custom-card">
            <h3 class="card-title">🚨 Masalah Bisnis (Business Problem)</h3>
            <p style="color: #cbd5e1; line-height: 1.8; text-align: justify;">
                Perusahaan <strong>Jaya Jaya Maju</strong> menghadapi tantangan serius berupa tingginya tingkat turnover (attrition) karyawan yang melampaui angka <strong>10%</strong>. 
                Tingkat turnover yang tinggi ini tidak hanya mengganggu stabilitas operasional harian dan efisiensi tim, namun juga memicu kerugian finansial yang signifikan melalui:
            </p>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Biaya Rekrutmen & Onboarding:</strong> Biaya pemasangan lowongan, wawancara, dan pelatihan personil baru.</li>
                <li><strong>Hilangnya Pengetahuan Institusional:</strong> Karyawan senior membawa keahlian kritis saat meninggalkan perusahaan.</li>
                <li><strong>Penurunan Moral & Produktivitas:</strong> Anggota tim yang tersisa mengalami kelelahan akibat beban kerja berlebih (overwork).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="custom-card">
            <h3 class="card-title">🎯 Solusi & Tujuan Proyek</h3>
            <p style="color: #cbd5e1; line-height: 1.8;">
                Melalui proyek ini, dibangun solusi terpadu berbasis <em>Machine Learning</em>:
            </p>
            <ol style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Dashboard Eksploratif:</strong> Memantau metrik attrition berdasarkan departemen, peran, dan faktor demografis.</li>
                <li><strong>Model Prediksi Dini:</strong> Mengidentifikasi karyawan berisiko tinggi mengundurkan diri sebelum keputusan resmi diambil.</li>
                <li><strong>Actionable Retention Insights:</strong> Memberikan panduan rekomendasi tindakan nyata bagi divisi HR untuk intervensi preventif.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 5. HALAMAN 2: DASHBOARD ANALITIK (LOOKER STUDIO)
# ==============================================================================
elif menu == "📊 Dashboard Analitik (Looker)":
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-tag">Exploratory Dashboard</span>
        <h1 class="hero-title">Visualisasi Data & Attrition Drivers</h1>
        <p class="hero-subtitle">
            Dashboard interaktif Looker Studio untuk meninjau pola, demografi, dan korelasi utama penyebab attrition karyawan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    looker_url = "https://lookerstudio.google.com/embed/reporting/e5923401-bf39-4e14-b3d5-ef4848897beb/page/0OxYF"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 0.8rem;">
        <a href="{looker_url.replace('/embed', '')}" target="_blank" style="text-decoration:none;">
            <button style="
                background: #ffb347; 
                color: #111; 
                border: none; 
                padding: 0.45rem 1.2rem; 
                border-radius: 8px; 
                font-weight: 600; 
                cursor: pointer;">
                🔗 Buka Dashboard Fullscreen
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Embed Looker Studio iframe
    if hasattr(st, "iframe"):
        st.iframe(looker_url, height=720, scrolling=True)
    elif hasattr(st.components.v1, "iframe"):
        st.components.v1.iframe(looker_url, height=720, scrolling=True)
    else:
        st.components.v1.html(
            f"""
            <iframe 
                src="{looker_url}" 
                width="100%" 
                height="720px" 
                style="border: 1px solid rgba(255, 179, 71, 0.2); border-radius: 12px; box-shadow: 0 6px 25px rgba(0,0,0,0.5);"
                allowfullscreen 
                sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
            </iframe>
            """,
            height=740
        )

    st.markdown("""
    <div class="custom-card" style="margin-top: 1.5rem;">
        <h4 style="color:#ffb347; margin-top:0;">💡 Temuan Kunci Analisis Data (EDA Insights):</h4>
        <ul style="color: #cbd5e1; line-height: 1.8; margin-bottom:0;">
            <li><strong>Lembur (OverTime):</strong> Karyawan yang rutin lembur memiliki probabilitas attrition hingga 3x lipat dibanding yang tidak lembur.</li>
            <li><strong>Job Level & Monthly Income:</strong> Attrition terkonsentrasi pada level pemula (Entry/Job Level 1) dengan kompensasi rendah.</li>
            <li><strong>Work-Life Balance & Jarak Rumah:</strong> Jarak tempuh > 15 km yang disertai work-life balance buruk berkontribusi besar terhadap keputusan resign.</li>
            <li><strong>Masa Kerja (YearsAtCompany):</strong> Titik kritis turnover terjadi pada rentang tahun ke-1 hingga ke-3 masa kerja.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. HALAMAN 3: PREDIKSI ATTRITION INDIVIDU
# ==============================================================================
elif menu == "🔮 Prediksi Attrition (Individu)":
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-tag">Predictive Model</span>
        <h1 class="hero-title">Simulasi & Prediksi Attrition Karyawan</h1>
        <p class="hero-subtitle">
            Masukkan parameter profil karyawan di bawah untuk memprediksi kecenderungan resign serta melihat tingkat probabilitas risikonya.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Profiles untuk kemudahan evaluasi
    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    with col_p1:
        if st.button("🚨 Muat Profil Risiko Tinggi"):
            st.session_state["age"] = 28
            st.session_state["distance"] = 25
            st.session_state["env_sat"] = 1
            st.session_state["job_level"] = 1
            st.session_state["monthly_income"] = 2600
            st.session_state["job_sat"] = 1
            st.session_state["job_inv"] = 1
            st.session_state["num_comp"] = 4
            st.session_state["rel_sat"] = 2
            st.session_state["total_work_years"] = 3
            st.session_state["work_life"] = 1
            st.session_state["years_company"] = 1
            st.session_state["years_role"] = 1
            st.session_state["years_manager"] = 0
            st.session_state["stock"] = 0
            st.session_state["job_role"] = "Sales Executive"
            st.session_state["marital_status"] = "Single"
            st.session_state["overtime"] = "Yes"
            st.rerun()

    with col_p2:
        if st.button("🟢 Muat Profil Stabil (Bertahan)"):
            st.session_state["age"] = 45
            st.session_state["distance"] = 4
            st.session_state["env_sat"] = 4
            st.session_state["job_level"] = 4
            st.session_state["monthly_income"] = 14000
            st.session_state["job_sat"] = 4
            st.session_state["job_inv"] = 3
            st.session_state["num_comp"] = 2
            st.session_state["rel_sat"] = 4
            st.session_state["total_work_years"] = 20
            st.session_state["work_life"] = 3
            st.session_state["years_company"] = 12
            st.session_state["years_role"] = 8
            st.session_state["years_manager"] = 7
            st.session_state["stock"] = 2
            st.session_state["job_role"] = "Manager"
            st.session_state["marital_status"] = "Married"
            st.session_state["overtime"] = "No"
            st.rerun()

    with col_p3:
        if st.button("🔄 Reset ke Nilai Rata-rata"):
            for key in ["age", "distance", "env_sat", "job_level", "monthly_income", "job_sat", 
                        "job_inv", "num_comp", "rel_sat", "total_work_years", "work_life", 
                        "years_company", "years_role", "years_manager", "stock", "job_role", 
                        "marital_status", "overtime"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Form Input Terstruktur dalam 3 Kolom
    with st.form("attrition_form"):
        col_form1, col_form2, col_form3 = st.columns(3)

        # ------------------ KOLOM 1: DATA PRIBADI & DEMOGRAFI ------------------
        with col_form1:
            st.markdown("#### 👤 Demografi & Pribadi")
            
            age = st.slider(
                "Umur (Age):", 
                min_value=18, max_value=65, 
                value=st.session_state.get("age", 34)
            )
            
            marital_options = ["Single", "Married", "Divorced"]
            default_marital = st.session_state.get("marital_status", "Married")
            marital_status = st.selectbox(
                "Status Pernikahan (Marital Status):",
                marital_options,
                index=marital_options.index(default_marital) if default_marital in marital_options else 0
            )

            distance = st.slider(
                "Jarak Rumah ke Kantor (km):", 
                min_value=1, max_value=50, 
                value=st.session_state.get("distance", 9)
            )

            num_comp = st.number_input(
                "Jumlah Perusahaan Sebelumnya:", 
                min_value=0, max_value=15, 
                value=st.session_state.get("num_comp", 2)
            )

            stock_level = st.selectbox(
                "Tingkat Opsi Saham (Stock Option Level):",
                [0, 1, 2, 3],
                index=st.session_state.get("stock", 1)
            )

        # ------------------ KOLOM 2: KARIR & JABATAN ------------------
        with col_form2:
            st.markdown("#### 💼 Karir & Jabatan")

            job_role_options = [
                'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                'Manager', 'Sales Representative', 'Research Director', 
                'Human Resources', 'Healthcare Representative', 'Manufacturing Director'
            ]
            default_role = st.session_state.get("job_role", "Sales Executive")
            job_role = st.selectbox(
                "Posisi / Peran Pekerjaan (Job Role):",
                job_role_options,
                index=job_role_options.index(default_role) if default_role in job_role_options else 0
            )

            job_level = st.select_slider(
                "Tingkat Jabatan (Job Level 1-5):",
                options=[1, 2, 3, 4, 5],
                value=st.session_state.get("job_level", 2)
            )

            monthly_income = st.number_input(
                "Pendapatan Bulanan ($ / Nominal):",
                min_value=1000, max_value=50000, step=250,
                value=st.session_state.get("monthly_income", 5000)
            )

            total_work_years = st.slider(
                "Total Tahun Bekerja (Total Working Years):",
                min_value=0, max_value=45,
                value=st.session_state.get("total_work_years", 10)
            )

            years_company = st.slider(
                "Lama di Perusahaan Ini (Years at Company):",
                min_value=0, max_value=40,
                value=st.session_state.get("years_company", 5)
            )

            years_role = st.slider(
                "Lama di Posisi Saat Ini (Years in Current Role):",
                min_value=0, max_value=20,
                value=st.session_state.get("years_role", 3)
            )

            years_manager = st.slider(
                "Lama Bersama Manajer Saat Ini (Years with Current Manager):",
                min_value=0, max_value=20,
                value=st.session_state.get("years_manager", 3)
            )

        # ------------------ KOLOM 3: KEPUASAN & WORK-LIFE ------------------
        with col_form3:
            st.markdown("#### 🌟 Kepuasan & Work-Life")

            overtime_options = ["No", "Yes"]
            default_ot = st.session_state.get("overtime", "No")
            overtime = st.radio(
                "Kerja Lembur Rutin (OverTime):",
                overtime_options,
                index=overtime_options.index(default_ot) if default_ot in overtime_options else 0,
                horizontal=True
            )

            env_sat = st.select_slider(
                "Kepuasan Lingkungan Kerja (1 = Rendah, 4 = Tinggi):",
                options=[1, 2, 3, 4],
                value=st.session_state.get("env_sat", 3)
            )

            job_sat = st.select_slider(
                "Kepuasan Pekerjaan (1 = Rendah, 4 = Tinggi):",
                options=[1, 2, 3, 4],
                value=st.session_state.get("job_sat", 3)
            )

            job_inv = st.select_slider(
                "Keterlibatan Kerja (Job Involvement 1-4):",
                options=[1, 2, 3, 4],
                value=st.session_state.get("job_inv", 3)
            )

            rel_sat = st.select_slider(
                "Kepuasan Hubungan Kerja (Relationship Sat 1-4):",
                options=[1, 2, 3, 4],
                value=st.session_state.get("rel_sat", 3)
            )

            work_life = st.select_slider(
                "Keseimbangan Kerja-Hidup (Work Life Balance 1-4):",
                options=[1, 2, 3, 4],
                value=st.session_state.get("work_life", 3)
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "⚡ Jalankan Prediksi Attrition", 
            use_container_width=True, 
            type="primary"
        )

    # ------------------ PROSES HASIL PREDIKSI ------------------
    if submitted:
        if model is None:
            st.error("Model belum berhasil dimuat. Periksa konfigurasi.")
        else:
            input_payload = {
                "Age": int(age),
                "DistanceFromHome": int(distance),
                "EnvironmentSatisfaction": int(env_sat),
                "JobLevel": int(job_level),
                "MonthlyIncome": int(monthly_income),
                "JobSatisfaction": int(job_sat),
                "JobInvolvement": int(job_inv),
                "NumCompaniesWorked": int(num_comp),
                "RelationshipSatisfaction": int(rel_sat),
                "TotalWorkingYears": int(total_work_years),
                "WorkLifeBalance": int(work_life),
                "YearsAtCompany": int(years_company),
                "YearsInCurrentRole": int(years_role),
                "YearsWithCurrManager": int(years_manager),
                "StockOptionLevel": int(stock_level),
                "JobRole": job_role,
                "MaritalStatus": marital_status,
                "OverTime": overtime
            }

            with st.spinner("Menganalisis probabilitas attrition..."):
                pred, probas = predict_proba_from_input(model, input_payload)

            p_attrition = probas["Attrition"] if probas else (1.0 if pred == 1 else 0.0)
            p_stay = probas["Stay"] if probas else (1.0 - p_attrition)
            risk_pct = round(p_attrition * 100, 1)

            st.markdown("---")
            st.markdown("### 📋 Hasil Analisis Prediksi")

            res_c1, res_c2 = st.columns([1.2, 1])

            with res_c1:
                if pred == 1:
                    st.markdown(f"""
                    <div class="badge-attrition-yes">
                        <h1 style="color: #ff4d4d; font-size: 3.5rem; margin:0;">🚨</h1>
                        <h2 style="color: #ff4d4d; margin: 0.5rem 0;">Karyawan Berpotensi Resign!</h2>
                        <p style="color: #ffffff; font-size: 1.15rem; margin-bottom: 0.5rem;">
                            Prediksi: <strong>YES (Attrition)</strong>
                        </p>
                        <p style="color: #cbd5e1; font-size: 0.95rem;">
                            Individu ini memiliki indikasi kuat untuk meninggalkan perusahaan dalam jangka pendek/menengah.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="badge-attrition-no">
                        <h1 style="color: #00e676; font-size: 3.5rem; margin:0;">✅</h1>
                        <h2 style="color: #00e676; margin: 0.5rem 0;">Karyawan Cenderung Bertahan</h2>
                        <p style="color: #ffffff; font-size: 1.15rem; margin-bottom: 0.5rem;">
                            Prediksi: <strong>NO (Loyal / Stay)</strong>
                        </p>
                        <p style="color: #cbd5e1; font-size: 0.95rem;">
                            Individu ini menunjukkan tingkat kepuasan dan stabilitas kerja yang memadai.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_c2:
                st.markdown("""
                <div class="custom-card">
                    <h4 style="color:#ffb347; margin-top:0;">📊 Tingkat Risiko Attrition</h4>
                """, unsafe_allow_html=True)

                st.progress(p_attrition)
                st.write(f"Probabilitas Resign: **{risk_pct}%** | Probabilitas Bertahan: **{round(p_stay * 100, 1)}%**")

                # Identifikasi pemicu risiko
                risk_factors = []
                if overtime == "Yes":
                    risk_factors.append("⚠️ Kerja Lembur Rutin (OverTime: Yes)")
                if env_sat <= 2:
                    risk_factors.append("⚠️ Kepuasan Lingkungan Kerja Rendah (Skor ≤ 2)")
                if job_sat <= 2:
                    risk_factors.append("⚠️ Kepuasan Pekerjaan Rendah (Skor ≤ 2)")
                if work_life <= 2:
                    risk_factors.append("⚠️ Keseimbangan Kerja-Hidup Kurang (Skor ≤ 2)")
                if distance >= 20:
                    risk_factors.append("⚠️ Jarak Tempuh Rumah Jauh (≥ 20 km)")
                if monthly_income < 3500 and job_level <= 2:
                    risk_factors.append("⚠️ Tingkat Kompensasi di Bawah Rata-Rata")

                if risk_factors:
                    st.markdown("**Faktor Pemicu Risiko Terdeteksi:**")
                    for rf in risk_factors:
                        st.markdown(f"<span style='color: #f87171;'>{rf}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("🟢 *Tidak ditemukan faktor pemicu risiko ekstrim.*")

                st.markdown("</div>", unsafe_allow_html=True)

            # Rekomendasi HR
            st.markdown("""
            <div class="custom-card" style="margin-top: 1rem;">
                <h4 style="color: #ffb347; margin-top:0;">💡 Rekomendasi Intervensi HR & Manajemen:</h4>
            """, unsafe_allow_html=True)

            if pred == 1:
                st.markdown("""
                1. **Evaluasi Beban Lembur:** Tinjau ulang jam kerja dan pertimbangkan pembagian tugas tim atau kompensasi insentif ekstra jika lembur tidak dapat dihindari.
                2. **Sesi 1-on-1 Empatik:** Jadwalkan diskusi empat mata bersama People Partner/HR untuk mendengarkan keluhan terkait lingkungan atau hubungan kerja.
                3. **Review Jalur Karir & Kompensasi:** Pastikan jenjang promosi dan penyesuaian gaji sesuai dengan kontribusi nyata yang diberikan.
                4. **Fleksibilitas Kerja:** Jika jarak rumah jauh, berikan opsi *hybrid working* (WFA/WFH parsial) untuk menekan kelelahan perjalanan.
                """)
            else:
                st.markdown("""
                1. **Pertahankan Keterlibatan Positif:** Berikan apresiasi dan pengakuan (recognition) atas dedikasi dan kinerja yang konsisten.
                2. **Pemberian Tanggung Jawab Baru:** Pertimbangkan mentoring atau keterlibatan dalam proyek strategis guna menjaga antusiasme jangka panjang.
                3. **Pertahankan Work-Life Balance:** Jaga ritme kerja kondusif agar komitmen karyawan tetap terjaga.
                """)
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 7. HALAMAN 4: PREDIKSI BATCH (UPLOAD CSV)
# ==============================================================================
elif menu == "📁 Prediksi Batch (Upload CSV)":
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-tag">Batch Processing</span>
        <h1 class="hero-title">Prediksi Attrition Massal (Upload CSV)</h1>
        <p class="hero-subtitle">
            Unggah file data karyawan dalam format CSV atau Excel untuk melakukan screening dan prediksi risiko turnover secara simultan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_b1, c_b2 = st.columns([1.5, 1])

    with c_b1:
        uploaded_file = st.file_uploader(
            "Pilih File CSV / Excel Karyawan:",
            type=["csv", "xlsx"]
        )

    with c_b2:
        st.markdown("""
        <div class="custom-card">
            <h5 style="color:#ffb347; margin-top:0;">📥 Unduh Template Contoh</h5>
            <p style="color:#cbd5e1; font-size:0.88rem;">Gunakan format template yang sudah disesuaikan agar prediksi berjalan akurat.</p>
        """, unsafe_allow_html=True)

        # File template
        template_path = "sample_data_attrition.csv"
        if os.path.exists(template_path):
            with open(template_path, "rb") as f:
                st.download_button(
                    label="📄 Unduh sample_data_attrition.csv",
                    data=f,
                    file_name="sample_data_attrition.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.success(f"✅ Berhasil memuat data: **{len(df_upload)} baris karyawan** terdeteksi.")
            
            with st.expander("👀 Tinjau 5 Data Teratas"):
                st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("🚀 Proses Prediksi Batch untuk Semua Karyawan", type="primary", use_container_width=True):
                if model is None:
                    st.error("Model Machine Learning belum aktif.")
                else:
                    with st.spinner("Memproses prediksi machine learning untuk seluruh baris data..."):
                        preds, probas = predict_from_dataframe(model, df_upload)

                        df_result = df_upload.copy()
                        df_result["Prediksi_Attrition"] = ["YES (Resign)" if p == 1 else "NO (Stay)" for p in preds]
                        if probas is not None:
                            df_result["Probabilitas_Resign_%"] = [round(pr * 100, 1) for pr in probas]

                    total_emp = len(df_result)
                    total_resign = sum(preds)
                    total_stay = total_emp - total_resign
                    pct_turnover = round((total_resign / total_emp) * 100, 1)

                    st.markdown("---")
                    st.markdown("### 📊 Ringkasan Hasil Screening Batch")

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Karyawan Dianalisis", f"{total_emp} orang")
                    with m2:
                        st.metric("Prediksi Resign (Yes)", f"{total_resign} orang", delta=f"{pct_turnover}%", delta_color="inverse")
                    with m3:
                        st.metric("Prediksi Bertahan (No)", f"{total_stay} orang")
                    with m4:
                        st.metric("Estimasi Attrition Rate", f"{pct_turnover}%", delta="Target < 10%", delta_color="off")

                    st.markdown("#### Tabel Lengkap Hasil Prediksi:")
                    
                    # Berikan warna highlight jika ada kolom Prediksi_Attrition
                    st.dataframe(
                        df_result, 
                        use_container_width=True
                    )

                    # Export button
                    csv_export = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Hasil Prediksi (CSV)",
                        data=csv_export,
                        file_name="hasil_prediksi_attrition_karyawan.csv",
                        mime="text/csv",
                        type="primary"
                    )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file atau melakukan prediksi: {e}")


# ==============================================================================
# 8. HALAMAN 5: INFO MODEL & ARSITEKTUR
# ==============================================================================
elif menu == "ℹ️ Info Model & Arsitektur":
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-tag">System Specs</span>
        <h1 class="hero-title">Arsitektur Model & Pipeline Deployment</h1>
        <p class="hero-subtitle">
            Spesifikasi teknis machine learning pipeline, tracking DagsHub MLflow, dan integrasi cloud.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color:#ffb347; margin-top:0;">🧠 Arsitektur Pipeline (Scikit-Learn)</h4>
            <table style="width:100%; color:#cbd5e1; border-collapse: collapse; font-size:0.92rem;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Estimator:</strong></td>
                    <td><code>RandomForestClassifier(random_state=42)</code></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Fitur Numerik:</strong></td>
                    <td>15 Fitur (StandardScaler)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Fitur Kategorik:</strong></td>
                    <td>JobRole, MaritalStatus, OverTime (OneHotEncoder)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Target Variabel:</strong></td>
                    <td>Attrition (0: No, 1: Yes)</td>
                </tr>
                <tr>
                    <td style="padding: 0.6rem 0;"><strong>Artifact Storage:</strong></td>
                    <td><code>model/rf_pipeline.pkl</code> (~1.5 MB)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color:#ffb347; margin-top:0;">☁️ Tracking & Deployment Registry</h4>
            <table style="width:100%; color:#cbd5e1; border-collapse: collapse; font-size:0.92rem;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>MLOps Server:</strong></td>
                    <td>MLflow Tracking di DagsHub</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Repository:</strong></td>
                    <td><code>AnggaBagusS/attrition-app</code></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Registered Model:</strong></td>
                    <td><code>models:/attrition_model/7</code></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <td style="padding: 0.6rem 0;"><strong>Frontend Target:</strong></td>
                    <td>Streamlit Community Cloud (Python 3.10+)</td>
                </tr>
                <tr>
                    <td style="padding: 0.6rem 0;"><strong>Dual Compatibility:</strong></td>
                    <td>Mendukung Streamlit & Flask Legacy API</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card">
        <h4 style="color:#ffb347; margin-top:0;">🚀 Cara Deploy ke Streamlit Community Cloud:</h4>
        <ol style="color:#cbd5e1; line-height:1.9;">
            <li>Pastikan repository ini sudah di-push ke <strong>GitHub</strong> Anda (pastikan folder <code>model/rf_pipeline.pkl</code> dan <code>requirements.txt</code> terunggah).</li>
            <li>Buka <a href="https://share.streamlit.io" target="_blank" style="color:#ffb347;">share.streamlit.io</a> dan login dengan akun GitHub Anda.</li>
            <li>Klik tombol <strong>"New app"</strong>.</li>
            <li>Pilih repository Anda, branch <code>main</code> (atau <code>master</code>).</li>
            <li>Pada kolom <strong>Main file path</strong>, masukkan <code>streamlit_app.py</code> atau <code>app.py</code>.</li>
            <li>Klik <strong>Deploy!</strong> Aplikasi Anda akan aktif dan dapat diakses publik secara online dalam hitungan menit.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
