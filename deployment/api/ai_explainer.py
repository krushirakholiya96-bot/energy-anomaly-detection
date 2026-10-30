from openai import OpenAI
import os

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = "llama-3.3-70b-versatile"


class AnomalyExplainer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def explain(self, sensor_data: dict, anomaly_score: float, top_sensor: str) -> str:
        prompt = f"""Industrial pump anomaly detected.
Anomaly Score: {anomaly_score:.3f}
Top Responsible Sensor: {top_sensor}
Readings:
- Current: {sensor_data['current']} A
- Voltage: {sensor_data['voltage']} V
- Pressure: {sensor_data['pressure']}
- Temperature: {sensor_data['temperature']} C
- Thermocouple: {sensor_data['thermocouple']} C
- Accelerometer: {sensor_data['accelerometer']}

In 2-3 lines explain for maintenance engineer:
1. What could be the problem?
2. Which sensor to check?
3. What immediate action to take?"""

        # Pehle MODEL try karo, fail ho to FALLBACK_MODEL, phir bhi fail ho to message
        for model in (MODEL, FALLBACK_MODEL):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    timeout=20,
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"AI explanation failed with model {model}: {e}")

        return "AI explanation abhi available nahi hai."
           
