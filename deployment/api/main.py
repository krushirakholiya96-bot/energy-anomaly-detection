from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from . import schemas, models, database
from .predictor import AnomalyPredictor
from .ai_explainer import AnomalyExplainer
from datetime import datetime

app = FastAPI(title="Energy Anomaly Detection API", version="1.0")
predictor = AnomalyPredictor()
explainer = AnomalyExplainer()

models.Base.metadata.create_all(bind=database.engine)

@app.get("/health")
def health():
    return {"status": "ok", "model": "v1.0"}

@app.post("/predict", response_model=schemas.PredictionOutput)
def predict(data: schemas.SensorInput, db: Session = Depends(database.get_db)):
    values = [data.current, data.voltage, data.pressure,
              data.temperature, data.thermocouple, data.accelerometer]
    result = predictor.predict(values)
    ai_text = None
    if result["is_anomaly"]:
        ai_text = explainer.explain(data.dict(), result["anomaly_score"], result["top_sensor"])
    db_pred = models.Prediction(**data.dict(), **result, ai_explanation=ai_text, timestamp=datetime.utcnow())
    db.add(db_pred)
    db.commit()
    return {**result, "ai_explanation": ai_text, "timestamp": datetime.utcnow(), "message": "ANOMALY ALERT!" if result["is_anomaly"] else "Normal"}

@app.get("/history")
def history(limit: int = 50, db: Session = Depends(database.get_db)):
    return db.query(models.Prediction).order_by(models.Prediction.timestamp.desc()).limit(limit).all()

@app.get("/stats")
def stats(db: Session = Depends(database.get_db)):
    total = db.query(models.Prediction).count()
    anomalies = db.query(models.Prediction).filter(models.Prediction.is_anomaly == True).count()
    return {"total": total, "anomalies": anomalies, "anomaly_rate": round(anomalies / total * 100, 1) if total > 0 else 0}