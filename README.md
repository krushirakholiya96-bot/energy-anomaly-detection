# ⚡ Energy Anomaly Detection

**Industrial pump anomaly detection — from raw sensor data to a deployed, AI-explained production system.**

An end-to-end machine learning system that detects anomalies in industrial pump sensor data *without labeled training data*, bridges unsupervised learning to a fast supervised model via pseudo-labeling, explains every alert with SHAP + an LLM, and ships as a containerized, cloud-deployed API + dashboard.

**Tech Stack:** Python · PyTorch · scikit-learn · XGBoost · SHAP · MLflow · UMAP · FastAPI · PostgreSQL · SQLAlchemy · Streamlit · Docker · Render · GitHub Actions · Groq LLM

---

## 🔗 Live Demo

| Service | Link |
|---|---|
| **API (Swagger Docs)** | [energy-anomaly-detection-u19f.onrender.com/docs](https://energy-anomaly-detection-u19f.onrender.com/docs) |
| **Dashboard** | [Streamlit App](https://energy-anomaly-detection-jywhyuymkgifcslpjw9ath.streamlit.app/) |

> Note: hosted on free tiers — the API may take 30–60s to wake up on the first request.

---

## 🧠 The Problem

Industrial pump failures are expensive and hard to predict. In the real world, you almost never have **labeled anomaly data** — engineers don't know in advance when a pump will fail, so a purely supervised approach isn't realistic on day one.

This project solves that with a workflow that mirrors how anomaly detection is actually built in industry:

```
Unsupervised Detection  →  Pseudo-Labeling  →  Supervised Model  →  Explainability  →  Deployment
   (Isolation Forest,         (score-based           (XGBoost)         (SHAP)         (FastAPI +
    LSTM Autoencoder)          thresholding)                                          Docker + Cloud)
```

**Why this matters:** it's not just "train a model." It demonstrates the judgment to work *without* labels, validate that judgment against ground truth, and then productionize the result — the actual lifecycle a company would ask for.

---

## 📊 Dataset

[**SKAB**](https://github.com/waico/SKAB) (Skoltech Anomaly Benchmark) — a real industrial pump dataset from Skoltech University, widely cited in anomaly-detection research. Per-second readings across 8 sensors (current, voltage, pressure, temperature, vibration, flow rate) with ground-truth anomaly and changepoint labels reserved strictly for evaluation.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌─────────────┐
│   SKAB      │────▶│  Isolation Forest │────▶│               │     │             │
│   Sensor    │     │  (baseline)       │     │  Pseudo-Label │────▶│  XGBoost /  │
│   Data      │────▶│  LSTM Autoencoder │────▶│  Generation   │     │  Random     │
│             │     │  (reconstruction  │     │  (threshold   │     │  Forest     │
│             │     │   error)          │     │   on scores)  │     │             │
└─────────────┘     └──────────────────┘     └───────────────┘     └──────┬──────┘
                                                                            │
                            ┌───────────────────────────────────────────────┘
                            ▼
                 ┌─────────────────┐    ┌────────────┐    ┌──────────────┐
                 │  SHAP            │───▶│  FastAPI   │───▶│  Streamlit   │
                 │  Explainability  │    │  + Groq LLM│    │  Dashboard   │
                 │                  │    │  + Postgres│    │  (3 pages)   │
                 └─────────────────┘    └────────────┘    └──────────────┘
```

---

## 🔬 Modeling Approach

| Stage | Technique | Purpose |
|---|---|---|
| Baseline | **Isolation Forest** | Fast classical anomaly scoring, no assumptions about data distribution |
| Deep Learning | **LSTM Autoencoder** | Learns normal temporal sensor patterns; anomalies = high reconstruction error |
| Bridge | **Pseudo-labeling** | Converts unsupervised reconstruction error into training labels (`error > threshold → anomaly`) |
| Supervised | **XGBoost / Random Forest** | Trained on pseudo-labels — faster inference, more interpretable, production-friendly |
| Explainability | **SHAP** | Identifies *which sensor* drove each anomaly (TreeExplainer for XGBoost, GradientExplainer for LSTM) |
| Visualization | **UMAP + Plotly** | 2D projection of the LSTM latent space — visually confirms normal/anomaly separation |
| Tracking | **MLflow** | Every experiment, hyperparameter set, and model version logged and compared |

**Threshold selection** for pseudo-labeling was compared across three methods — 95th percentile, 3-sigma rule, and F1-optimal — with the F1-optimal threshold chosen for the best precision/recall balance.

---

## 🚀 Deployment

| Layer | Technology |
|---|---|
| ML Serving | XGBoost + StandardScaler (`.pkl`), FastAPI + Pydantic |
| AI Explanation | Groq LLM (Llama 3) — generates a plain-language root-cause explanation per alert |
| Database | PostgreSQL (Neon) via SQLAlchemy ORM — every prediction + explanation is persisted |
| Dashboard | Streamlit — Live Prediction / History & Analytics / AI Insights |
| Containerization | Docker + docker-compose |
| Cloud | Render (API) + Streamlit Cloud (dashboard) |
| CI/CD | GitHub Actions — auto-deploy on push to `main` |

### API Endpoints
- `POST /predict` — submit sensor readings, get an anomaly verdict + AI explanation
- `GET /history` — recent predictions
- `GET /stats` — anomaly rate and summary stats
- `GET /health` — service status

---

## 📁 Project Structure

```
energy-anomaly-detection/
├── notebooks/              # 01–11: data loading → EDA → preprocessing → windows
│                            #        → Isolation Forest → LSTM → pseudo-labels
│                            #        → XGBoost → evaluation → SHAP → UMAP
├── models/                 # isolation_forest.pkl, best_lstm_model.pth,
│                            # supervised_model.pkl, scaler.pkl
├── reports/                 # evaluation_report.md, business_insights.md
├── visualizations/          # SHAP plots, UMAP interactive HTML
├── deployment/
│   ├── api/                 # FastAPI: main.py, predictor.py, ai_explainer.py,
│   │                         # database.py, models.py, schemas.py
│   ├── dashboard/            # Streamlit app.py (3 pages)
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── .github/workflows/       # deploy.yml — CI/CD
└── requirements.txt
```

---

## 🛠️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<krushirakholiya96-bot>/energy-anomaly-detection.git
cd energy-anomaly-detection

# 2. Set up environment variables
cp .env.example .env
# fill in DATABASE_URL and GROQ_API_KEY

# 3. Install dependencies
pip install -r deployment/api/requirements.txt
pip install -r deployment/dashboard/requirements.txt

# 4. Run the API
cd deployment/api
uvicorn main:app --reload

# 5. Run the dashboard (in a separate terminal)
cd deployment/dashboard
streamlit run app.py
```

**Or run everything with Docker:**
```bash
docker-compose up --build
```

---

## 🎯 Design Decisions

- **Why start unsupervised?** Real production environments rarely have labeled anomalies on day one — this mirrors the actual constraint.
- **Why LSTM specifically?** Sensor readings are sequential; an LSTM autoencoder captures temporal dependencies that a static model would miss.
- **Why add a supervised model on top?** Interpretability and inference speed. Once pseudo-labels exist, a tree-based model is faster to serve and easier to explain than reconstruction error alone.
- **Why SHAP?** "An anomaly was detected" isn't actionable on its own — a maintenance engineer needs to know *which sensor* to check. SHAP quantifies that.
- **Why MLflow?** Reproducibility across dozens of hyperparameter runs (LSTM tuning alone spans 5+ combinations).

---

## ⚠️ Limitations & Future Work

- Pseudo-labels are an approximation of ground truth — they were validated against SKAB's actual labels, but some noise is expected in any pseudo-labeling pipeline.
- Currently tuned and evaluated on the SKAB `valve1` subset; generalizing to other SKAB subsystems or a different sensor set would need re-validation.
- Planned: online/streaming inference, model retraining trigger based on drift detection, and authentication on the API.

