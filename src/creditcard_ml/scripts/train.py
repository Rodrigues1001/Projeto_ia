import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
import joblib

def main():
    print("🚀 Training model...")

    df = pd.read_csv("data/processed.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)

    # save metrics
    with open("metrics.json", "w") as f:
        json.dump(report, f, indent=4)

    # save model
    joblib.dump(model, "src/creditcard_ml/model/model.pkl")

    print("📊 metrics.json saved")
    print("🤖 model.pkl saved")

if __name__ == "__main__":
    main()
