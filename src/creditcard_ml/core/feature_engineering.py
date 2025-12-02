import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

SCALER_AMOUNT_PATH = "src/creditcard_ml/model/scaler_amount.pkl"
SCALER_TIME_PATH = "src/creditcard_ml/model/scaler_time.pkl"

def build_features(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    df = df.copy()

    if is_train:
        raise RuntimeError(
            "build_features() não é usado durante o treino — o preprocess do DVC cuida disso."
        )

    # carregar scalers já treinados
    scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
    scaler_time = joblib.load(SCALER_TIME_PATH)

    df["scaled_amount"] = scaler_amount.transform(df[["Amount"]])
    df["scaled_time"] = scaler_time.transform(df[["Time"]])

    return df
