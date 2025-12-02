import joblib

MODEL_PATH = "src/creditcard_ml/model/model.pkl"
SCALER_PATH = "src/creditcard_ml/model/scaler.pkl"

def load_model():
  return joblib.load(MODEL_PATH)


def load_scaler():
  return joblib.load(SCALER_PATH)
