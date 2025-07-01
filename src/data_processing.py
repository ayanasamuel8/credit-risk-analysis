import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
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

# -----------------------------
# Custom Transformer for RFM Features
# -----------------------------
class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

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

        # RFM Scaling and Clustering
        rfm_scaled = rfm.drop(columns=["CustomerId"]).fillna(0)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_scaled)

        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

        # Assign high-risk label to cluster with lowest Frequency and AvgAmount
        cluster_profiles = rfm.groupby("Cluster")[["Frequency", "AvgAmount"]].mean()
        high_risk_cluster = cluster_profiles.sum(axis=1).idxmin()
        rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

        X_ = X_.merge(rfm[["CustomerId", "Recency", "Frequency", "AvgAmount", "AmountStdDev", "is_high_risk"]], on="CustomerId", how="left")
        return X_

# -----------------------------
# Log Transformer
# -----------------------------
log_transformer = FunctionTransformer(np.log1p, validate=True)

# -----------------------------
# Pipeline Assembly Function
# -----------------------------
def create_feature_pipeline():
    log_scaled_cols = ['Amount', 'Value', 'Frequency', 'AvgAmount', 'AmountStdDev', 'Recency']
    numeric_only = []
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    numerical_log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", log_transformer),
        ("scaler", StandardScaler())
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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
