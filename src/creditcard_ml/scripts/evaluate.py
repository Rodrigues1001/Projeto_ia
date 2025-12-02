import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
import json

def main():
    print("📈 Evaluating model...")

    df = pd.read_csv("data/processed.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    model = joblib.load("src/creditcard_ml/model/model.pkl")

    proba = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, proba)

    with open("eval.json", "w") as f:
        json.dump({"auc": auc}, f, indent=4)

    print("📄 eval.json saved")

if __name__ == "__main__":
    main()
