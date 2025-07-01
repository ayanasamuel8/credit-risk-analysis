import mlflow.pyfunc
import pandas as pd
from src.data_processing import create_feature_pipeline
import joblib

def predict(df: pd.DataFrame, model_name: str ="CreditRisk_xgboost_model", stage: str ="Production"):
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    data = df.copy()
    pipeline = joblib.load("../artifacts/fitted_pipeline.pkl")
    transformed_data = pipeline.transform(data)
    # 🏷️ Build full column list
    cat_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"]\
        .named_steps["onehot"].get_feature_names_out(['ProductCategory', 'ChannelId', 'PricingStrategy'])

    all_cols = ['Amount', 'Value', 'Frequency', 'AvgAmount', 'AmountStdDev', 'Recency'] + list(cat_cols)

    # 🧾 Create transformed DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=all_cols)
    predictions = model.predict(transformed_df)
    data["is_high_risk"] = predictions

    return data