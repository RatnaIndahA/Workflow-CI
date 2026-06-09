# inference.py
import os
import time
import glob
import psutil
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge, generate_latest, REGISTRY
import threading

app = Flask(__name__)

# Prometheus metrics from flask_exporter (otomatis untuk request & latency)
metrics = PrometheusMetrics(app)

# Custom metrics untuk prediksi per kelas
prediction_counter = Counter('prediction_class_total', 'Total predictions by class', ['pred_class'])

# Custom metrics untuk RAM dan CPU (Gauge agar bisa di-update periodik)
ram_usage = Gauge('system_ram_usage_bytes', 'RAM usage in bytes')
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percent')

# Daftar kemungkinan path relatif (disesuaikan dengan lokasi file)
possible_paths = [
    "../../MLProject/model.pkl",      # dari folder 7.inference.py ke model
    "../MLProject/model.pkl",         # jika dari folder Monitoring_dan_Logging langsung
    "model.pkl",                      # lokal
]

MODEL_PATH = None
for path in possible_paths:
    full_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(full_path):
        MODEL_PATH = full_path
        break

if MODEL_PATH is None:
    raise FileNotFoundError("Model tidak ditemukan. Cek path: " + str(possible_paths))

model = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# Fungsi untuk update metrik sistem secara periodik
def update_system_metrics():
    while True:
        ram_usage.set(psutil.virtual_memory().used)
        cpu_usage.set(psutil.cpu_percent(interval=1))
        time.sleep(5)  # update setiap 5 detik

# Jalankan thread background untuk update sistem metrics
thread = threading.Thread(target=update_system_metrics, daemon=True)
thread.start()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        # Pastikan kolom sesuai dengan feature names model
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0].max()
        
        # Update prediction counter berdasarkan kelas
        prediction_counter.labels(pred_class=str(pred)).inc()
        
        return jsonify({"prediction": str(pred), "confidence": float(proba)})
    except Exception as e:
        # Error akan tetap tercatat oleh PrometheusMetrics
        return jsonify({"error": str(e)}), 500

# Endpoint untuk expose metrics ke Prometheus (sudah otomatis oleh PrometheusMetrics di /metrics)
# Namun kita perlu pastikan metrics sistem juga terekspos. PrometheusMetrics akan menambahkan /metrics sendiri.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)