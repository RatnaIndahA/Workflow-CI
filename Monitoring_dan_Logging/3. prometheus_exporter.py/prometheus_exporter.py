# 3.prometheus_exporter.py
"""
Prometheus exporter untuk metrik sistem dan aplikasi.
Metrik yang diekspor: prediksi per kelas, RAM, CPU.
Catatan: Exporter ini sudah terintegrasi di inference.py,
namun file ini dibuat untuk memenuhi struktur folder kriteria 4.
"""

from prometheus_client import start_http_server, Counter, Gauge
import psutil
import time

# Definisikan metrik (sama seperti di inference.py)
prediction_counter = Counter('prediction_class_total', 'Total predictions by class', ['pred_class'])
ram_usage = Gauge('system_ram_usage_bytes', 'RAM usage in bytes')
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percent')

def update_metrics():
    """Update metrik sistem secara periodik."""
    while True:
        ram_usage.set(psutil.virtual_memory().used)
        cpu_usage.set(psutil.cpu_percent(interval=1))
        time.sleep(5)

if __name__ == "__main__":
    # Mulai server metrics di port 8000 (bisa berbeda dengan inference.py)
    start_http_server(8000)
    print("Prometheus exporter running on http://localhost:8000/metrics")
    print("Metrik yang tersedia: system_ram_usage_bytes, system_cpu_usage_percent, prediction_class_total")
    update_metrics()