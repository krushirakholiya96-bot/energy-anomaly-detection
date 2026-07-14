
# ⚡ Energy Anomaly Detection

Industrial pump anomaly detection system using ML + AI.

## Live Demo
- API: https://energy-anomaly-detection-u19f.onrender.com/docs
- Dashboard: https://energy-anomaly-detection-jywhyuymkgifcslpjw9ath.streamlit.app/

## Tech Stack
- FastAPI + PostgreSQL (Neon)
- XGBoost + LSTM Autoencoder
- Groq LLM (AI Explanation)
- Streamlit Dashboard
- Docker + Render + GitHub Actions

## Results
- Models: Isolation Forest, LSTM Autoencoder, XGBoost
- End-to-end ML deployment pipeline

## How to Run
1. Clone the repo
2. Add .env file with DATABASE_URL and GROQ_API_KEY
3. pip install -r deployment/api/requirements.txt
4. uvicorn main:app --reload
