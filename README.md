# 🏦 Credit Risk Model

## 📊 Credit Scoring Business Understanding

### 1. Impact of Basel II on Model Interpretability

The Basel II Capital Accord emphasizes that financial institutions must actively measure and manage their exposure to credit risk using sound, quantitative methods. This includes maintaining transparency, explainability, and auditability in their credit scoring systems. 

In practice, this means any predictive model used to assess borrower creditworthiness must be interpretable. Financial regulators and internal compliance teams must be able to understand and justify why a model made a specific decision—especially in cases of loan rejection. As a result, our model must not only be accurate, but also well-documented and explainable. Clear variable definitions, interpretable feature effects (e.g., via Weight of Evidence or SHAP values), and consistent behavior across time are critical for regulatory acceptance and trust in the model.

---

### 2. Why a Proxy Target Is Necessary and the Risks Involved

The provided dataset lacks a direct "default" label, meaning we do not have ground truth data that tells us whether a customer failed to repay a loan. To solve this, we must define a **proxy target variable**—a stand-in indicator of high-risk behavior.

In this project, we use RFM (Recency, Frequency, Monetary) analysis to segment customers based on their engagement patterns. Customers who transact infrequently, spend less, and haven’t transacted recently may be grouped into a "high-risk" cluster, which we label as 1. Others are labeled 0 (low risk).

However, relying on a proxy introduces business risks:
- **False Positives**: We may misclassify good customers as high-risk, leading to lost business opportunities.
- **False Negatives**: We may approve credit for customers who will default, increasing financial losses.
- **Bias**: The proxy may reflect engagement behavior more than actual financial behavior, introducing bias against certain customer groups.

Hence, our proxy must be defined with business logic and carefully evaluated to avoid long-term damage to customer trust and bank profitability.

---

### 3. Trade-offs Between Interpretable and Complex Models

In credit risk modeling—especially within regulated financial environments—there is often a trade-off between interpretability and predictive performance.

**Simple models**, such as Logistic Regression combined with Weight of Evidence (WoE) encoding, offer:
- High transparency: it's easy to explain decisions.
- Regulatory friendliness: easier to audit and justify to compliance teams.
- Robustness: lower risk of overfitting and easier monitoring in production.

**Complex models**, such as Gradient Boosting Machines (GBM), provide:
- Higher predictive power: can capture non-linear patterns in the data.
- Better handling of variable interactions and missing data.

However, complex models are often seen as "black boxes" and may require additional tools like SHAP or LIME to provide post-hoc explanations. These explanations, while helpful, may not always meet regulatory requirements for auditability.

In regulated settings like banking, it's often best to start with an interpretable baseline model. Once trust and performance are established, more complex models can be introduced with sufficient documentation and explainability tools.

---

## 📈 Exploratory Data Analysis (EDA) Summary

We performed initial data exploration and uncovered the following key insights:

- `CountryCode` has only one unique value across all records, so it does not provide any discriminatory power and can be safely removed.
- Both `Amount` and `Value` are highly right-skewed due to a few very large transactions. To make them more suitable for modeling, a **log transformation** (`log1p`) is planned.
- `PricingStrategy` appears fairly balanced and symmetric across categories, so it will be retained and encoded appropriately (e.g., using one-hot encoding).
- Analysis of `ProductCategory` against our proxy target `is_high_risk` revealed that categories such as `airtime` and `financial_services` dominate in transaction volume and have a **significant share of high-risk users**. These insights can guide risk policy and feature creation.

We will continue expanding on these insights as we progress with feature engineering and modeling.

---

## 🧠 Project Structure

```graphql
credict-risk-analysis/
│
├── artifacts/                 # Saved trained pipeline and metadata
│   └── fitted_pipeline.pkl
│
├── docker-compose.yml        # Docker orchestration file
├── Dockerfile                # Docker build config for the API service
│
├── mlruns/                   # MLflow experiment tracking files
│
├── notebooks/                # Experiment notebooks
│   ├── 1.0-eda.ipynb
│   ├── 2.0-feature_enginering.ipynb
│   ├── 3.0-model_training.ipynb
│   └── 4.0-prediction_checking.ipynb
│   └── mlruns/               # (sometimes duplicated under notebooks/)
│
├── requirements.txt          # Project dependencies
│
├── src/                      # Core logic and service code
│   ├── data_processing.py    # Feature engineering pipeline
│   ├── train.py              # Model training and MLflow registration
│   ├── predict.py            # Prediction logic for inference
│   ├── utils.py              # Utility functions (e.g. logging, config)
│   └── api/                  # FastAPI application
│       ├── main.py           # API entry point
│       └── pydantic_models.py # Request/response schemas
│
├── tests/                    # Unit tests for data and model code
│   ├── test_data_processing.py
│   └── test_train.py
│
└── README.md
```

---

## 🚀 Quick Start

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
```
# ✅ 2. Run Unit Tests
```bash
pytest
```
# 🧪 3. Train the Model and Register It

```bash
python src/train.py
```
### This will:

Log the trained model in the `mlruns/` directory using MLflow

Save the fitted pipeline to `artifacts/fitted_pipeline.pkl`

# 🧬 4. Run the API Server
Start with FastAPI:
```bash
uvicorn src.api.main:app --reload
```
Or using Docker:
```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```
Or with Docker Compose:
```bash
docker-compose up
```
Then open: http://localhost:8000/docs for the Swagger API interface.