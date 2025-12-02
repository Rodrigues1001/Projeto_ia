import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

from creditcard_ml.data.loader import load_data
from creditcard_ml.core.feature_engineering import build_features

MODEL_PATH = "src/creditcard_ml/model/model.pkl"


def train_model():
    df = load_data()
    df = build_features(df, is_train=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"AUC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


def main():
    train_model()
