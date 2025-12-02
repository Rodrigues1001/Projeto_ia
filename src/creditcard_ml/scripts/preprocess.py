import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = "data/creditcard.csv"
PROCESSED_PATH = "data/processed.csv"

SCALER_AMOUNT_PATH = "src/creditcard_ml/model/scaler_amount.pkl"
SCALER_TIME_PATH = "src/creditcard_ml/model/scaler_time.pkl"

def preprocess():
    print("🔧 Preprocessing dataset...")

    df = pd.read_csv(RAW_PATH)

    # Separar features e target
    y = df["Class"]
    X = df.drop(columns=["Class"])

    # Criar scalers individuais
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    X["scaled_amount"] = scaler_amount.fit_transform(X[["Amount"]])
    X["scaled_time"] = scaler_time.fit_transform(X[["Time"]])

    # Salvar scalers
    joblib.dump(scaler_amount, SCALER_AMOUNT_PATH)
    joblib.dump(scaler_time, SCALER_TIME_PATH)

    # NÃO remover Amount e Time (você quer manter)
    df_final = X.copy()
    df_final["Class"] = y

    df_final.to_csv(PROCESSED_PATH, index=False)

    print(f"✅ Saved processed dataset: {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()
