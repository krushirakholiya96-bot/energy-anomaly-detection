\# ⚡ Energy Anomaly Detection



Industrial pump anomaly detection system using ML + AI.



\## Live Demo

\- API: https://your-app.onrender.com

\- Dashboard: https://your-app.streamlit.app



\## Tech Stack

\- FastAPI + PostgreSQL (Neon)

\- XGBoost + LSTM Autoencoder

\- Groq LLM (AI Explanation)

\- Streamlit Dashboard

\- Docker + Render + GitHub Actions



\## Results

\- AUC-ROC: 0.XX

\- Models: Isolation Forest, LSTM Autoencoder, XGBoost



\## How to Run Locally

1\. Clone the repo

2\. Add .env file with DATABASE\_URL and GROQ\_API\_KEY

3\. pip install -r deployment/api/requirements.txt

4\. uvicorn deployment.api.main:app --reload

