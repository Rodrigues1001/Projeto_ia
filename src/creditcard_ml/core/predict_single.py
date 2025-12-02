import pandas as pd

from creditcard_ml.core.model_loader import load_model
from creditcard_ml.core.feature_engineering import build_features

def predict_single(data: dict) -> dict:
  df = pd.DataFrame([data])
  df = build_features(df, is_train=False)

  model = load_model()
  prob = model.predict_proba(df)[0][1]

  return {"fraud_probability": float(prob)}
