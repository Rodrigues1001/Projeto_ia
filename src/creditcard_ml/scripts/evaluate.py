import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import json

PROCESSED_PATH = "data/processed.csv"
MODEL_PATH = "src/creditcard_ml/model/model.pkl"
EVAL_PATH = "eval.json"

def evaluate():
    print("📈 Evaluating model...")

    df = pd.read_csv(PROCESSED_PATH)

    y = df["Class"]
    X = df.drop(columns=["Class"])

    model = joblib.load(MODEL_PATH)

    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    with open(EVAL_PATH, "w") as f:
        json.dump({"AUC": auc}, f, indent=2)

    print(f"📄 eval.json saved with AUC={auc:.4f}")

if __name__ == "__main__":
    evaluate()
