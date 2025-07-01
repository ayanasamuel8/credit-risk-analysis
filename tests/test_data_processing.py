import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import create_feature_pipeline

def generate_mock_data():
    np.random.seed(42)
    now = datetime.now()

    n = 10
    data = {
        "TransactionId": [f"TransactionId_{i}" for i in range(n)],
        "BatchId": [f"BatchId_{i}" for i in range(n)],
        "AccountId": [f"AccountId_{i%4}" for i in range(n)],
        "SubscriptionId": [f"SubscriptionId_{i%3}" for i in range(n)],
        "CustomerId": [i % 4 + 1 for i in range(n)],
        "CurrencyCode": ["UGX"] * n,
        "CountryCode": [256] * n,
        "ProviderId": [f"ProviderId_{i%5}" for i in range(n)],
        "ProductId": [f"ProductId_{i%6}" for i in range(n)],
        "ProductCategory": ["airtime", "data", "sms", "airtime", "data", "sms", "airtime", "data", "sms", "airtime"],
        "ChannelId": [f"ChannelId_{i%3}" for i in range(n)],
        "Amount": np.random.randint(100, 1000, size=n),
        "Value": np.random.randint(50, 500, size=n),
        "TransactionStartTime": [(now - timedelta(days=i)).isoformat() for i in range(n)],
        "PricingStrategy": [str(i % 3) for i in range(n)],
        "FraudResult": np.random.randint(0, 2, size=n)
    }

    return pd.DataFrame(data)

def test_pipeline_runs_and_adds_is_high_risk():
    df = generate_mock_data()
    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(df)

    # Check row count
    assert transformed.shape[0] == df.shape[0], "Transformed rows must match input"

    # Ensure RFM features are added
    rfm_df = pipeline.named_steps["rfm_features"].transform(df)
    assert "is_high_risk" in rfm_df.columns, "'is_high_risk' should be in RFM output"
    assert rfm_df["is_high_risk"].isin([0, 1]).all(), "is_high_risk must be binary"

    # No NaNs in transformed result
    assert not np.isnan(transformed).any(), "Transformed data contains NaNs"

    print("✅ Pipeline test passed successfully.")
