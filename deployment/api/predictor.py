import joblib
import numpy as np
import os

class AnomalyPredictor:
    def __init__(self):
        base_path = os.path.join(os.path.dirname(__file__), 'models')
        self.model = joblib.load(os.path.join(base_path, 'supervised_model.pkl'))
        self.feature_names = [
            'current', 'voltage', 'pressure',
            'temperature', 'thermocouple', 'accelerometer'
        ]

    def predict(self, values: list) -> dict:
        arr = np.array(values).reshape(1, -1)

        score = self.model.predict_proba(arr)[0][1]
        is_anomaly = bool(score > 0.5)

        importances = self.model.feature_importances_
        top_idx = np.argmax(importances)
        top_sensor = self.feature_names[top_idx]

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': round(float(score), 4),
            'top_sensor': top_sensor
        }