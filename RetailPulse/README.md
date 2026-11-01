# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

A runnable starter project for retail analytics. Includes sales analytics, a machine-learning demand forecast, customer segmentation, churn-risk scoring, inventory recommendations, and a Streamlit dashboard.

## 1. Requirements
- Python 3.10 or 3.11
- VS Code or another Python editor

## 2. Run locally
```bash
# Windows
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## 3. Project structure
```text
RetailPulse/
├── app.py                  # Streamlit dashboard
├── data/
│   ├── sales.csv           # Synthetic daily sales sample
│   └── customers.csv       # Synthetic customer features + churn label
├── src/analytics.py        # Forecast, segmentation, churn, inventory logic
├── models/                 # Reserved for persisted models
├── reports/                # Reserved for generated reports
├── tests/                  # Tests
├── Dockerfile
└── requirements.txt
```

## 4. What is implemented
- Sales KPIs and product-level charts
- Random Forest demand forecast using lag, rolling-average and weekday features
- K-Means customer segmentation with readable segment labels
- Random Forest churn-risk model and holdout ROC-AUC
- Inventory reorder suggestion using recent average demand and a safety factor
- Streamlit interactive dashboard
- Docker image definition

## 5. Important notes
The included CSV files are **synthetic demo data**, not real retailer data. The churn label is simulated to make the workflow demonstrable. Validate labels and model quality on real, representative data before business use. Forecasting currently uses a Random Forest baseline; Prophet and LSTM are possible later extensions, not silently assumed to be installed. Kubernetes, MLflow tracking, CI/CD, streaming ETL, and production drift alerts are roadmap items, not yet deployed.

## 6. Next production milestones
1. Replace sample CSVs with a documented source schema and data validation.
2. Add time-based backtesting (MAE, RMSE, MAPE) and forecast intervals.
3. Persist models and metrics with MLflow.
4. Add data/model drift monitoring and alert thresholds.
5. Containerize and deploy with Kubernetes.
6. Add authentication, secrets management, logging, tests, and CI/CD.

## 7. Run tests
```bash
pytest -q
```
