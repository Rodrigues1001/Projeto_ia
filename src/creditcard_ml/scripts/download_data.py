import kagglehub
import pandas as pd
import os

def main():
    print("⬇️ Baixando dataset do Kaggle com kagglehub...")

    dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = f"{dataset_path}/creditcard.csv"

    # onde vamos salvar para o pipeline
    output_path = "data/creditcard.csv"

    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(csv_path)
    df.to_csv(output_path, index=False)

    print(f"📁 Dataset salvo em: {output_path}")

if __name__ == "__main__":
    main()
