# Proyek MLOps: Prediksi Harga Saham Astra

## Gambaran Umum Proyek

Proyek ini merupakan **implementasi end-to-end Machine Learning Operations (MLOps)** untuk memprediksi **harga saham PT Astra International Tbk (ASII.JK)** menggunakan data historis.

Fokus utama proyek ini **bukan pada kompleksitas model**, melainkan pada **workflow MLOps**, mulai dari data ingestion, preprocessing, training, deployment, hingga monitoring sistem machine learning.

Proyek ini dibuat untuk memenuhi **tugas Project MLOps – Program Studi Sains Data ITERA 2025**.

---

## Tujuan Proyek

* Memahami workflow lengkap MLOps
* Menerapkan praktik terbaik MLOps (reproducibility, automation, CI/CD)
* Mengimplementasikan sistem machine learning yang dapat dideploy secara online
* Melatih kerja kolaboratif dalam pengembangan sistem ML berbasis GitHub

---

## Ruang Lingkup Proyek

* **Sumber Data:** Data historis harga saham Astra (ASII.JK)
* **Tugas Prediksi:** Prediksi harga penutupan saham hari berikutnya (regresi)
* **Fokus Utama:**

  * Data ingestion & preprocessing
  * Experiment tracking (MLflow)
  * Model serving berbasis REST API
  * CI/CD dengan GitHub Actions
  * Monitoring sistem ML

---

## Arsitektur Sistem

```
Sumber Data → Data Ingestion → Preprocessing → Training (MLflow)
                                         ↓
                                  Model Registry
                                         ↓
                                 API Inference (FastAPI)
                                         ↓
                                 Docker & Deployment
```

---

## Struktur Repository

```
mlops-astra-prediction/
├── data/
│   ├── raw/                # Data mentah (DVC tracked)
│   └── processed/          # Data hasil preprocessing
├── src/
│   ├── data/               # Script data ingestion & preprocessing
│   ├── models/             # Script training & evaluasi model
│   └── serving/            # API inference (FastAPI)
├── experiments/            # Notebook EDA & eksperimen
├── tests/                  # Unit testing
├── .github/workflows/      # CI/CD (GitHub Actions)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.10
* **Machine Learning:** Scikit-learn
* **Experiment Tracking:** MLflow
* **Data Versioning:** DVC
* **API Framework:** FastAPI
* **Containerization:** Docker
* **CI/CD:** GitHub Actions
* **Monitoring:** Prometheus (basic metrics)

---

## Cara Menjalankan Proyek (Local)

### 1️⃣ Install Dependency

```bash
pip install -r requirements.txt
```

### 2️⃣ Data Ingestion

```bash
python src/data/fetch_data.py
```

### 3️⃣ Data Preprocessing

```bash
python src/data/preprocess.py
```

### 4️⃣ Training Model

```bash
python src/models/train.py
```

Menjalankan MLflow UI:

```bash
mlflow ui
```

### 5️⃣ Menjalankan API Inference

```bash
uvicorn src.serving.app:app --reload --port 8080
```

Endpoint prediksi:

```http
POST http://localhost:8080/predict
```

---

## Menjalankan dengan Docker

Build image:

```bash
docker build -t mlops-astra .
```

Run container:

```bash
docker run -p 8080:8080 mlops-astra
```

---

## 🔄 CI/CD Pipeline

Pipeline CI/CD berjalan otomatis menggunakan **GitHub Actions** dengan tahapan:

1. Install dependency
2. Menjalankan unit test
3. Build Docker image
4. Push image ke Docker Registry

Konfigurasi pipeline terdapat pada:

```
.github/workflows/ci.yml
```

---

## Monitoring

* Monitoring jumlah request dan latency API
* Logging error dan performa model
* Metric diekspos menggunakan Prometheus

---

## 👥Pembagian Tugas Tim

| Nama Anggota | NIM       | Peran                        | Tanggung Jawab                                                                            |
| ------------ | --------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| **Salwa**   | **NIM 1** | **Data Engineer**            | Data ingestion, preprocessing data, pembuatan fitur, data versioning menggunakan DVC      |
| **Tria Yunanni**   | **NIM 2** | **ML Engineer**              | Training model, evaluasi model, experiment tracking menggunakan MLflow                    |
| **Meira**   | **NIM 3** | **MLOps Engineer**           | Pembuatan API inference (FastAPI), containerization menggunakan Docker, deployment sistem |
| **Chalifia Wananda**   | **122450076** | **DevOps / Project Manager** | CI/CD menggunakan GitHub Actions, monitoring sistem, dokumentasi proyek & koordinasi tim  |


---

