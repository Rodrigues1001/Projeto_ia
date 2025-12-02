import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

PROCESSED_PATH = "data/processed.csv"
MODEL_PATH = "src/creditcard_ml/model/model.pkl"
METRICS_PATH = "metrics.json"
import json

def train():
    print("🚀 Training model...")

    df = pd.read_csv(PROCESSED_PATH)

    y = df["Class"]
    X = df.drop(columns=["Class"])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Salvar modelo
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "n_features": X.shape[1],
        "features": list(X.columns),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("📊 Model trained and saved!")
    print(f"🤖 Model path: {MODEL_PATH}")

if __name__ == "__main__":
    train()
