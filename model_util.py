import os
import joblib
import pandas as pd

# === 1. Load model dari lokal atau DagsHub ===
def get_model():
    os.makedirs("model", exist_ok=True)
    local_path = "model/rf_pipeline.pkl"

    # Prioritaskan model lokal jika sudah tersedia
    if os.path.exists(local_path):
        try:
            print("✅ Model lokal ditemukan, menggunakan model lokal.")
            return joblib.load(local_path)
        except Exception as e:
            print(f"⚠️ Gagal memuat model lokal: {e}")

    # Jika tidak ada lokal, coba unduh dari DagsHub menggunakan MLflow
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.exceptions import MlflowException

        uri_artifacts = "https://dagshub.com/AnggaBagusS/attrition-app.mlflow"
        mlflow.set_tracking_uri(uri_artifacts)
        model_uri = "models:/attrition_model/7"  # versi model sesuai DagsHub

        print("🔄 Mengambil model dari DagsHub...")
        model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(model, local_path)
        print(f"✅ Model berhasil disimpan di lokal: {local_path}")
        return model
    except Exception as e:
        print(f"⚠️ Gagal mengambil model dari DagsHub: {e}")
        if os.path.exists(local_path):
            print("🔁 Menggunakan model lokal yang sudah ada.")
            return joblib.load(local_path)
        else:
            raise RuntimeError(f"❌ Tidak ada model lokal atau remote yang bisa digunakan: {e}")


# === 2. Urutan fitur sesuai training (pipeline internal akan handle encoding) ===
expected_cols = [
    "JobRole", "MaritalStatus", "OverTime",
    "Age", "DistanceFromHome", "EnvironmentSatisfaction", "JobLevel",
    "MonthlyIncome", "JobSatisfaction", "JobInvolvement", "NumCompaniesWorked",
    "RelationshipSatisfaction", "TotalWorkingYears",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsWithCurrManager", "StockOptionLevel"
]


# === 3. Fungsi prediksi untuk single input ===
def predict_from_input(model, input_dict):
    df_input = pd.DataFrame([input_dict])

    # Pastikan semua kolom lengkap dan urut sesuai training
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = None

    df_input = df_input[expected_cols]

    try:
        y_pred = model.predict(df_input)
        return int(y_pred[0])
    except Exception as e:
        raise RuntimeError(f"Error saat prediksi: {e}")


# === 4. Fungsi prediksi probabilitas untuk single input ===
def predict_proba_from_input(model, input_dict):
    df_input = pd.DataFrame([input_dict])

    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = None

    df_input = df_input[expected_cols]

    try:
        pred = int(model.predict(df_input)[0])
        probas = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(df_input)[0]
            # p[0]: probability Stay (0), p[1]: probability Attrition (1)
            probas = {
                "Stay": float(p[0]),
                "Attrition": float(p[1])
            }
        return pred, probas
    except Exception as e:
        raise RuntimeError(f"Error saat prediksi probabilitas: {e}")


# === 5. Fungsi prediksi batch dari DataFrame ===
def predict_from_dataframe(model, df):
    df_clean = df.copy()

    for col in expected_cols:
        if col not in df_clean.columns:
            df_clean[col] = None

    df_clean = df_clean[expected_cols]

    try:
        preds = model.predict(df_clean)
        probas = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(df_clean)
            probas = [float(x[1]) for x in p]  # Probabilitas Attrition (class 1)
        return preds, probas
    except Exception as e:
        raise RuntimeError(f"Error saat prediksi batch: {e}")
