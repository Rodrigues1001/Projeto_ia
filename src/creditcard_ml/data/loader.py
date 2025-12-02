import kagglehub
import pandas as pd

def load_data():
  # Baixa o dataset e retorna o caminho local
  dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

  # O arquivo CSV está dentro do diretório baixado
  file_path = f"{dataset_path}/creditcard.csv"

  df = pd.read_csv(file_path)
  return df
