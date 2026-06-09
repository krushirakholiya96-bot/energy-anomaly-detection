from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SensorInput(BaseModel):
    current: float
    voltage: float
    pressure: float
    temperature: float
    thermocouple: float
    accelerometer: float

class PredictionOutput(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    top_sensor: str
    ai_explanation: Optional[str] = None
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True