import streamlit as st
import requests
import pandas as pd
import plotly.express as px

try:
    API_URL = st.secrets["API_URL"]
except:
    import os
    API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Energy Anomaly Detection", layout="wide")
st.title("⚡ Energy Anomaly Detection Dashboard")

page = st.sidebar.radio("Navigation",
    ["Live Prediction", "History & Analytics", "AI Insights"])

# ── PAGE 1: LIVE PREDICTION ──────────────────────────────────────────────
if page == "Live Prediction":
    st.subheader(" Enter Sensor Values ")
    c1, c2, c3 = st.columns(3)
    current       = c1.number_input("Current (A)",   value=0.5)
    voltage       = c2.number_input("Voltage (V)",   value=220.0)
    pressure      = c3.number_input("Pressure",      value=1.0)
    temperature   = c1.number_input("Temperature",   value=85.0)
    thermocouple  = c2.number_input("Thermocouple",  value=30.0)
    accelerometer = c3.number_input("Accelerometer", value=0.01)

    if st.button("Predict Now", type="primary"):
        with st.spinner("Predicting..."):
            res = requests.post(f"{API_URL}/predict", json={
                "current"      : current,
                "voltage"      : voltage,
                "pressure"     : pressure,
                "temperature"  : temperature,
                "thermocouple" : thermocouple,
                "accelerometer": accelerometer
            }, timeout=60)
            if res.status_code == 200:
                res = res.json()
            else:
                st.error(f"API Error: {res.status_code} - {res.text}")
                st.stop()

        if res["is_anomaly"]:
            st.error(f"🚨 ANOMALY DETECTED! Score: {res['anomaly_score']:.3f}")
            st.warning(f"Top Sensor: {res['top_sensor']}")
            if res.get("ai_explanation"):
                st.info(f"🤖 AI Analysis: {res['ai_explanation']}")
        else:
            st.success(f"✅ System Normal | Score: {res['anomaly_score']:.3f}")

# ── PAGE 2: HISTORY ──────────────────────────────────────────────────────
elif page == "History & Analytics":
    stats = requests.get(f"{API_URL}/stats", timeout=30).json()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", stats["total"])
    c2.metric("Anomalies Found",   stats["anomalies"])
    c3.metric("Anomaly Rate",      f"{stats['anomaly_rate']}%")

    df = pd.DataFrame(
        requests.get(f"{API_URL}/history?limit=100", timeout=30).json()
    )
    if not df.empty:
        st.plotly_chart(px.line(df, x="timestamp", y="anomaly_score",
                                color="is_anomaly",
                                title="Anomaly Score Over Time"))
        st.dataframe(df[["timestamp", "is_anomaly",
                          "anomaly_score", "top_sensor"]])

# ── PAGE 3: AI INSIGHTS ──────────────────────────────────────────────────
elif page == "AI Insights":
    st.subheader("AI Explanations — Last Anomalies")
    df = pd.DataFrame(
        requests.get(f"{API_URL}/history?limit=20", timeout=30).json()
    )
    if not df.empty:
        anomalies = df[df["is_anomaly"] == True]
        if anomalies.empty:
            st.info("Abhi tak koi anomaly detect nahi hui.")
        for _, row in anomalies.iterrows():
            with st.expander(
                f"{row['timestamp']} | Score: {row['anomaly_score']:.3f}"
            ):
                st.write(row.get("ai_explanation", "No AI explanation"))
