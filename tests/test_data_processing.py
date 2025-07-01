import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processing import create_feature_pipeline

def generate_mock_data():
    np.random.seed(42)
    now = datetime.now()
    data = {
        "TransactionId": range(1, 11),
        "CustomerId": [1, 2, 1, 3, 4, 4, 2, 3, 3, 1],
        "TransactionStartTime": [(now - timedelta(days=i)).isoformat() for i in range(10)],
        "Amount": np.random.randint(100, 1000, size=10),
        "Value": np.random.randint(50, 500, size=10),
        "ProductCategory": ["A", "B", "A", "C", "A", "B", "C", "B", "A", "C"],
        "ChannelId": ["online", "offline", "offline", "online", "offline", "offline", "online", "offline", "online", "online"],
        "PricingStrategy": ["discount", "premium", "standard", "discount", "standard", "premium", "discount", "standard", "premium", "discount"]
    }
    return pd.DataFrame(data)

def test_pipeline_runs_and_adds_is_high_risk():
    df = generate_mock_data()
    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(df)

    # Check shape consistency
    assert transformed.shape[0] == df.shape[0], "Transformed rows must match input"

    # Check if 'is_high_risk' added correctly
    rfm_df = pipeline.named_steps["rfm_features"].transform(df)
    assert "is_high_risk" in rfm_df.columns, "'is_high_risk' should be in RFM output"
    assert rfm_df["is_high_risk"].isin([0, 1]).all(), "is_high_risk must be binary"

    # Check no NaNs after transformation
    assert not np.isnan(transformed).any(), "Transformed data contains NaNs"

    print("✅ Pipeline test passed successfully.")
