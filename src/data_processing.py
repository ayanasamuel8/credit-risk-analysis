import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# -----------------------------
# Custom Transformer for DateTime
# -----------------------------
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.datetime_col] = pd.to_datetime(X_[self.datetime_col])
        X_["Hour"] = X_[self.datetime_col].dt.hour
        X_["DayOfWeek"] = X_[self.datetime_col].dt.dayofweek
        X_["Month"] = X_[self.datetime_col].dt.month
        return X_

class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=10):
        self.max_categories = max_categories
        self.encoders = {}
        self.feature_names_out = []

    def fit(self, X, y=None):
        self.encoders = {}
        for col in X.columns:
            n_unique = X[col].nunique()
            if n_unique <= self.max_categories:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            else:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        encoded = []
        self.feature_names_out = []
        for col, encoder in self.encoders.items():
            X_col = X[[col]]
            transformed = encoder.transform(X_col)
            encoded.append(transformed)

            # Store feature names for later (especially for one-hot)
            if isinstance(encoder, OneHotEncoder):
                names = encoder.get_feature_names_out([col])
            else:
                names = [col]
            self.feature_names_out.extend(names)

        return np.hstack(encoded)

    def get_feature_names_out(self):
        return self.feature_names_out


# -----------------------------
# Custom Transformer for RFM Features
# -----------------------------
class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_["TransactionStartTime"] = pd.to_datetime(X_["TransactionStartTime"])
        if not self.snapshot_date:
            self.snapshot_date = X_["TransactionStartTime"].max() + pd.Timedelta(days=1)

        rfm = X_.groupby("CustomerId").agg({
            "TransactionStartTime": lambda x: (self.snapshot_date - x.max()).days,
            "TransactionId": "count",
            "Amount": ["mean", "std"]
        })

        rfm.columns = ["Recency", "Frequency", "AvgAmount", "AmountStdDev"]
        rfm = rfm.reset_index()
        
        # Handle NaNs in AmountStdDev
        rfm["AmountStdDev"] = rfm["AmountStdDev"].fillna(0)

        # Scale features to 0-1
        features_to_scale = ["Recency", "Frequency", "AvgAmount", "AmountStdDev"]
        rfm_scaled = self.scaler.fit_transform(rfm[features_to_scale])

        # Compute risk score
        # Higher Recency => more risk (weight +1)
        # Lower Frequency => more risk (weight -1)
        # Lower AvgAmount => more risk (weight -1)
        # Higher AmountStdDev => maybe more risk (weight +0.5)
        risk_score = (
            rfm_scaled[:, 0]  # Recency
            - rfm_scaled[:, 1]  # Frequency
            - rfm_scaled[:, 2]  # AvgAmount
            + 0.5 * rfm_scaled[:, 3]  # AmountStdDev
        )

        # Normalize risk score to 0-1
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        rfm["risk_score"] = risk_score
        # Define high risk as top 20% risk scores (adjust threshold as needed)
        threshold = np.quantile(risk_score, 0.60)
        rfm["is_high_risk"] = (rfm["risk_score"] >= threshold).astype(int)

        X_ = X_.merge(rfm[["CustomerId", "Recency", "Frequency", "AvgAmount", "AmountStdDev", "is_high_risk"]], on="CustomerId", how="left")
        return X_


# -----------------------------
# Log Transformer
# -----------------------------
def symmetric_log_func(x):
    return np.sign(x) * np.log1p(np.abs(x))

symmetric_log_transformer = FunctionTransformer(
    func=symmetric_log_func,
    validate=True
)

# -----------------------------
# Pipeline Assembly Function
# -----------------------------
def create_feature_pipeline():
    log_scaled_cols = ['Amount', 'Value', 'Frequency', 'AvgAmount', 'AmountStdDev', 'Recency']
    numeric_only = []
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'SubscriptionId', 'ProviderId', 'ProductId']

    numerical_log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", symmetric_log_transformer),
        ("scaler", StandardScaler())
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("smart_encoder", SmartCategoricalEncoder(max_categories=10))
    ])


    preprocessor = ColumnTransformer(transformers=[
        ("num_log", numerical_log_pipeline, log_scaled_cols),
        ("num", numerical_pipeline, numeric_only),
        ("cat", categorical_pipeline, categorical_features)
    ])

    full_pipeline = Pipeline([
        ("rfm_features", RFMFeatureEngineer()),
        ("datetime_features", DateTimeFeatureExtractor()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/raw/data.csv")  # Update to original raw file
    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(df)

    # Get categorical feature names
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    cat_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"]\
        .named_steps["onehot"].get_feature_names_out(categorical_features)

    # Combine all feature names
    log_scaled_cols = ['Amount', 'Value', 'Frequency', 'AvgAmount', 'AmountStdDev']
    numeric_only = ['Recency']
    all_cols = list(log_scaled_cols) + numeric_only + list(cat_cols)

    rfm_df = pipeline.named_steps["rfm_features"].transform(df)
    transformed_df["is_high_risk"] = rfm_df["is_high_risk"].values

    print(transformed_df.head())
