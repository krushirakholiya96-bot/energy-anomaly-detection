from sqlalchemy import Column, Integer, Float, Boolean, DateTime, String, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime, default=datetime.utcnow)
    current        = Column(Float)
    voltage        = Column(Float)
    pressure       = Column(Float)
    temperature    = Column(Float)
    thermocouple   = Column(Float)
    accelerometer  = Column(Float)
    anomaly_score  = Column(Float)
    is_anomaly     = Column(Boolean)
    top_sensor     = Column(String)
    ai_explanation = Column(Text, nullable=True)
    model_version  = Column(String, default='v1.0')