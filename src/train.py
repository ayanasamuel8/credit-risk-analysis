import pandas as pd
import mlflow
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(X_train, y_train, model_name="logistic", tracking_uri="mlruns"):
    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit Risk Modeling")

    with mlflow.start_run() as run:
        if model_name == "logistic":
            model = LogisticRegression(max_iter=1000)
            params = {"C": [0.01, 0.1, 1, 10]}
        elif model_name == "random_forest":
            model = RandomForestClassifier()
            params = {"n_estimators": [100, 200], "max_depth": [5, 10]}
        elif model_name == "xgboost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            params = {"n_estimators": [100, 200], "max_depth": [3, 6]}
        else:
            raise ValueError("Unsupported model")

        grid = GridSearchCV(model, param_grid=params, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        mlflow.log_params(grid.best_params_)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        return best_model, run.info.run_id


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics

def save_model(model, path="models/best_model.pkl"):
    joblib.dump(model, path)
