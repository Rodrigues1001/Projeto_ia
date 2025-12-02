import pandas as pd

from creditcard_ml.core.model_loader import load_model
from creditcard_ml.core.feature_engineering import build_features

"""
Recebe um objeto com as features de um registro de cartã o de credito
e retorna as probabilidades de fraude calculadas com base nos dados do objeto.

Parameters:
  data (dict): Objeto com as features do registro de cartão de crédito.

Returns:
  dict: Dicionário contendo a probabilidade de fraude do registro.
"""
def predict_single(data: dict) -> dict:
  df = pd.DataFrame([data])
  df = build_features(df, is_train=False)

  model = load_model()
  prob = model.predict_proba(df)[0][1]

  return {"fraud_probability": float(prob)}
