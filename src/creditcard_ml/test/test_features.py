from creditcard_ml.data.loader import load_data
from creditcard_ml.core.feature_engineering import build_features

def test_feature_engineering():
  df = load_data().head(10)
  df_feat = build_features(df, is_train=True)
  assert df_feat is not None
