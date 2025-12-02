import pandas as pd
import typer

from creditcard_ml.core.model_loader import load_model
from creditcard_ml.core.feature_engineering import build_features

CHUNK_SIZE = 200_000
app = typer.Typer()


"""
Realiza previsões em lote com base em um conjunto de dados.

Parameters:
  input_path (str): Caminho para o arquivo de entrada (CSV).
  output_path (str): Caminho para o arquivo de saída (CSV).

Returns:
  str: Caminho para o arquivo de saída gerado.
"""
def predict_batch(input_path: str, output_path: str) -> str:
  model = load_model()

  open(output_path, "w").close()

  reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE)
  batch_num = 0

  for chunk in reader:
    batch_num += 1
    print(f"Processando chunk {batch_num}...")

    features = build_features(chunk.copy(), is_train=False)
    probs = model.predict_proba(features)[:, 1]

    chunk["fraud_probability"] = probs

    chunk.to_csv(
      output_path,
      mode="a",
      index=False,
      header=(batch_num == 1),
    )

  print(f"Processamento completo. Arquivo gerado: {output_path}")
  return output_path

"""
  Realiza previsões em lote com base em um conjunto de dados.

  Parameters:
    input (str): Caminho para o arquivo de entrada (CSV).
    output (str): Caminho para o arquivo de saída (CSV).

  Returns:
    str: Caminho para o arquivo de saída gerado.
"""
@app.command()
def cli(input: str, output: str):
  predict_batch(input, output)


if __name__ == "__main__":
  app()
