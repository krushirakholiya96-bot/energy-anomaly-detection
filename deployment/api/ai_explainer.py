from openai import OpenAI
import os

class AnomalyExplainer:
    def __init__(self):
        self.client = OpenAI(
            api_key  = os.getenv("GROQ_API_KEY"),
            base_url = "https://api.groq.com/openai/v1"
        )

    def explain(self, sensor_data: dict, anomaly_score: float, top_sensor: str) -> str:
        if not sensor_data.get("is_anomaly", False):
            return "System normal hai — koi action required nahi."

        prompt = f"""Industrial pump mein anomaly detect hui hai.
Anomaly Score: {anomaly_score:.3f}
Top Responsible Sensor: {top_sensor}
Readings:
- Current: {sensor_data['current']} A
- Voltage: {sensor_data['voltage']} V
- Pressure: {sensor_data['pressure']}
- Temperature: {sensor_data['temperature']} C
- Thermocouple: {sensor_data['thermocouple']} C
- Accelerometer: {sensor_data['accelerometer']}

Maintenance engineer ke liye 2-3 lines mein explain karo:
1. Kya problem ho sakti hai?
2. Kaunsa sensor check karna chahiye?
3. Immediate action kya lena chahiye?"""

        resp = self.client.chat.completions.create(
            model      = "llama3-8b-8192",
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 150
        )
        return resp.choices[0].message.content