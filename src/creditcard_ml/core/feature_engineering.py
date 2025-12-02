import pandas as pd
import joblib
import os

# Resolve caminho absoluto automaticamente, independentemente de onde o código está rodando
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_AMOUNT_PATH = os.path.join(BASE_DIR, "..", "model", "scaler_amount.pkl")
SCALER_TIME_PATH   = os.path.join(BASE_DIR, "..", "model", "scaler_time.pkl")

def build_features(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    df = df.copy()

    if is_train:
        raise RuntimeError(
            "build_features() não é usado durante o treino — o preprocess do DVC cuida disso."
        )

    # Normaliza caminhos
    scaler_amount_path = os.path.realpath(SCALER_AMOUNT_PATH)
    scaler_time_path   = os.path.realpath(SCALER_TIME_PATH)

    # Carrega scalers
    scaler_amount = joblib.load(scaler_amount_path)
    scaler_time   = joblib.load(scaler_time_path)

    # Aplica transformações
    df["scaled_amount"] = scaler_amount.transform(df[["Amount"]])
    df["scaled_time"]   = scaler_time.transform(df[["Time"]])

    return df
