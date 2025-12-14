# 📈 Proyek MLOps: Prediksi Harga Saham Astra (ASII.JK)

[![GitHub](https://img.shields.io/badge/GitHub-code-blue?style=flat\&logo=github\&logoColor=white)](https://github.com/prsdm/mlops-project)

Selamat datang di **Proyek Prediksi Harga Saham Astra**. Proyek ini bertujuan untuk membangun **pipeline MLOps end-to-end** untuk memprediksi harga penutupan saham **PT Astra International Tbk (ASII.JK)** menggunakan data historis.

Fokus utama proyek ini adalah pada **implementasi workflow MLOps**, bukan pada kompleksitas model machine learning.

---

## 🎯 Tujuan Proyek

* Menerapkan workflow MLOps secara lengkap
* Mengelola data, model, dan eksperimen secara terstruktur
* Melakukan deployment model ke dalam bentuk API
* Menerapkan CI/CD dan monitoring sistem ML

---

## 🏗️ Diagram Arsitektur

Diagram berikut menggambarkan alur sistem dari data ingestion hingga deployment model:

```
Sumber Data (Yahoo Finance)
        ↓
Data Ingestion
        ↓
Data Preprocessing
        ↓
Model Training & Experiment Tracking (MLflow)
        ↓
Model Registry
        ↓
API Inference (FastAPI)
        ↓
Docker & Deployment
        ↓
Monitoring & Logging
```

---

## 🚀 Cara Menjalankan Proyek

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<username>/mlops-astra-stock-prediction.git
cd mlops-astra-stock-prediction
```

---

### 2️⃣ Setup Environment

Pastikan Python 3.8+ sudah terinstall.

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3️⃣ Data Ingestion

Mengambil data historis saham Astra (ASII.JK) dari Yahoo Finance:

```bash
python src/data/fetch_data.py
```

---

### 4️⃣ Data Preprocessing

```bash
python src/data/preprocess.py
```

---

### 5️⃣ Training Model

Training model dan pencatatan eksperimen menggunakan MLflow:

```bash
python src/models/train.py
```

Menjalankan MLflow UI:

```bash
mlflow ui
```

---

### 6️⃣ Menjalankan FastAPI

```bash
uvicorn src.serving.app:app --reload --port 8080
```

Endpoint prediksi:

```http
POST http://localhost:8080/predict
```

---

### 7️⃣ Docker

Build Docker image:

```bash
docker build -t mlops-astra .
```

Run container:

```bash
docker run -p 8080:8080 mlops-astra
```

---

### 8️⃣ Monitoring Model

* Monitoring jumlah request dan latency API
* Logging performa sistem dan error
* Metrics diekspos menggunakan Prometheus

---

## 👥 Pembagian Tugas Tim

| Nama   | NIM   | Peran          | Tanggung Jawab                     |
| ------ | ----- | -------------- | ---------------------------------- |
| Salwa Farhanatussaidah | 122450011 | Data Engineer  | Data ingestion, preprocessing, DVC |
| Tria Yunanni | 122450062 | ML Engineer    | Training model, evaluasi, MLflow   |
| Meira Listyaningrum | 122450055 | MLOps Engineer | API, Docker, deployment            |
| Chalifia Wananda | 122450076 | DevOps / PM    | CI/CD, monitoring, dokumentasi     |

---

## 📜 Lisensi

MIT License
