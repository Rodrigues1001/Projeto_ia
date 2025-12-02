import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

SCALER_PATH = "src/creditcard_ml/model/scaler.pkl"

"""
Converte um dataframe em um outro com as features escaladas.

Parameters:
  df (pd.DataFrame): Dataframe com as features a serem escaladas.
  is_train (bool): Indica se o modelo deve ser treinado (True) ou se as features
    devem ser escaladas com base em um modelo pré-treinado (False).

Returns:
  pd.DataFrame: Dataframe com as features escaladas.
"""
def build_features(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
  if "Time" in df.columns:
    df = df.drop(columns=["Time"])

  if "Class" in df.columns:
    feature_cols = [c for c in df.columns if c not in ["Class"]]
  else:
    feature_cols = df.columns.tolist()

  if is_train:
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, SCALER_PATH)
  else:
    scaler = joblib.load(SCALER_PATH)
    df[feature_cols] = scaler.transform(df[feature_cols])

  return df
