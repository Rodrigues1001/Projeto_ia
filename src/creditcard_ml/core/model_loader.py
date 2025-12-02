import joblib

MODEL_PATH = "src/creditcard_ml/model/model.pkl"
SCALER_PATH = "src/creditcard_ml/model/scaler.pkl"

"""
Carrega o modelo treinado com base no caminho definido em MODEL_PATH.

Retorna:
  O modelo treinado carregado com base no caminho MODEL_PATH.
"""
def load_model():
  return joblib.load(MODEL_PATH)


"""
Carrega o scaler com base no caminho definido em SCALER_PATH.

Retorna:
  O scaler carregado com base no caminho SCALER_PATH.
"""
def load_scaler():
  return joblib.load(SCALER_PATH)
