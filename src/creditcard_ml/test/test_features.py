from creditcard_ml.data.loader import load_data
from creditcard_ml.core.feature_engineering import build_features

"""
  Testa se a feature engineering é realizada corretamente.

  Carrega o conjunto de dados de cartões de crédito,
  realiza a feature engineering e verifica se o dataframe
  resultante não é nulo.
"""
def test_feature_engineering():
  df = load_data().head(10)
  df_feat = build_features(df, is_train=True)
  assert df_feat is not None
