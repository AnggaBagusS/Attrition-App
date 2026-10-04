from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
from model_util import get_model, predict_from_input

app = Flask(__name__)

MODEL_PATH = "model/rf_pipeline.pkl"


# === 1. Fungsi untuk load model dari lokal atau DagsHub ===
def load_or_download_model():
    os.makedirs("model", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print("✅ Model lokal ditemukan, menggunakan model lokal.")
        model = joblib.load(MODEL_PATH)
    else:
        print("⚙️ Model lokal tidak ditemukan, mencoba mengunduh dari DagsHub...")
        model = get_model()
    return model


# === 2. ROUTE HOME ===
@app.route('/')
def home():
    return render_template('home.html')


# === 3. ROUTE DASHBOARD ===
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard_view.html')


# === 4. ROUTE MANUAL DOWNLOAD MODEL ===
@app.route('/model')
def download_model():
    model = get_model()
    if model:
        message = "✅ Model berhasil diunduh dari DagsHub dan disimpan di lokal."
    else:
        message = "❌ Gagal mengunduh model dari DagsHub."
    return render_template('home.html', message=message)


# === 5. ROUTE UNTUK FORM PREDIKSI ===
@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    model = load_or_download_model()
    prediction = None
    input_data = {}

    if request.method == "POST":
        try:
            input_data = {
                "Age": int(request.form.get("Age", 0)),
                "DistanceFromHome": int(request.form.get("DistanceFromHome", 0)),
                "EnvironmentSatisfaction": int(request.form.get("EnvironmentSatisfaction", 0)),
                "JobLevel": int(request.form.get("JobLevel", 0)),
                "MonthlyIncome": int(request.form.get("MonthlyIncome", 0)),
                "JobSatisfaction": int(request.form.get("JobSatisfaction", 0)),
                "JobInvolvement": int(request.form.get("JobInvolvement", 0)),
                "NumCompaniesWorked": int(request.form.get("NumCompaniesWorked", 0)),
                "RelationshipSatisfaction": int(request.form.get("RelationshipSatisfaction", 0)),
                "TotalWorkingYears": int(request.form.get("TotalWorkingYears", 0)),
                "WorkLifeBalance": int(request.form.get("WorkLifeBalance", 0)),
                "YearsAtCompany": int(request.form.get("YearsAtCompany", 0)),
                "YearsInCurrentRole": int(request.form.get("YearsInCurrentRole", 0)),
                "YearsWithCurrManager": int(request.form.get("YearsWithCurrManager", 0)),
                "StockOptionLevel": int(request.form.get("StockOptionLevel", 0)),
                "JobRole": request.form.get("JobRole", ""),
                "MaritalStatus": request.form.get("MaritalStatus", ""),
                "OverTime": request.form.get("OverTime", "")
            }

            result = predict_from_input(model, input_data)
            prediction = "Yes" if result == 1 else "No"

        except Exception as e:
            prediction = f"❌ Terjadi kesalahan saat prediksi: {str(e)}"

    return render_template("predict_view.html", features=input_data, prediction=prediction)


# === 6. ROUTE UNTUK API PREDIKSI (JSON) ===
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()

    input_data = {
        "Age": data.get("Age"),
        "DistanceFromHome": data.get("DistanceFromHome"),
        "EnvironmentSatisfaction": data.get("EnvironmentSatisfaction"),
        "JobLevel": data.get("JobLevel"),
        "MonthlyIncome": data.get("MonthlyIncome"),
        "JobSatisfaction": data.get("JobSatisfaction"),
        "JobInvolvement": data.get("JobInvolvement"),
        "NumCompaniesWorked": data.get("NumCompaniesWorked"),
        "RelationshipSatisfaction": data.get("RelationshipSatisfaction"),
        "TotalWorkingYears": data.get("TotalWorkingYears"),
        "WorkLifeBalance": data.get("WorkLifeBalance"),
        "YearsAtCompany": data.get("YearsAtCompany"),
        "YearsInCurrentRole": data.get("YearsInCurrentRole"),
        "YearsWithCurrManager": data.get("YearsWithCurrManager"),
        "StockOptionLevel": data.get("StockOptionLevel"),
        "JobRole": data.get("JobRole"),
        "MaritalStatus": data.get("MaritalStatus"),
        "OverTime": data.get("OverTime")
    }

    try:
        model = load_or_download_model()
        result = predict_from_input(model, input_data)
        prediction = "Yes" if result == 1 else "No"
        status = "success"
    except Exception as e:
        prediction = None
        status = f"error: {str(e)}"

    return jsonify({
        "status": status,
        "prediction": prediction,
        "input": input_data
    })


# === 7. MAIN APP ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
