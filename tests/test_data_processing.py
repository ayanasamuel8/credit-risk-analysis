import pandas as pd
from src.data_processing import clean_data, feature_engineering

def test_clean_data():
    df = pd.DataFrame({'a': [1, 2, None]})
    cleaned = clean_data(df)
    assert isinstance(cleaned, pd.DataFrame)

def test_feature_engineering():
    df = pd.DataFrame({'a': [1, 2, 3]})
    features = feature_engineering(df)
    assert isinstance(features, pd.DataFrame)
