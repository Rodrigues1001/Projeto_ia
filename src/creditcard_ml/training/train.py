import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/processed.csv"
MODEL_PATH = "src/creditcard_ml/model/model.pkl"
SCALER_AMOUNT_PATH = "src/creditcard_ml/model/scaler_amount.pkl"
SCALER_TIME_PATH = "src/creditcard_ml/model/scaler_time.pkl"

def main():
    print("🚀 Training model...")

    # 1. Carregar dataset já pré-processado pelo DVC
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2. Treino/validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced_subsample",
        random_state=42
    )

    model.fit(X_train, y_train)

    # 4. Avaliação
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "auc": float(roc_auc_score(y_val, probs)),
        "precision": float(precision_score(y_val, preds)),
        "recall": float(recall_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds)),
    }

    # 5. Salvar modelo
    joblib.dump(model, MODEL_PATH)
    print("🤖 model.pkl saved")

    # 6. Salvar métricas
    import json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("📊 metrics.json saved")


if __name__ == "__main__":
    main()
