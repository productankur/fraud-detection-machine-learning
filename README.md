
# 🔍 UPI Fraud Detection using Machine Learning

A real-time fraud detection system built on 6.3 million UPI transactions —
using dual specialist ML models, SHAP explainability, and a live Streamlit interface.

---

## 🚀 Live App

👉 [Launch Fraud Detection App](ahref-https://fraud-detection-machine-learning-fkbzgreypnrearys4nqsmg.streamlit.app/)

---

## 📌 What This Project Does

Financial fraud detection is a critical challenge — especially when only
0.13% of transactions are actually fraudulent. This project builds an
end-to-end ML pipeline that:

- Analyses 6.3 million transactions across 30 days
- Detects fraudulent TRANSFER and CASH_OUT transactions in real time
- Explains exactly WHY each transaction was flagged using SHAP
- Allows fraud analysts to upload a CSV and download flagged cases instantly

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Transactions | 6,362,620 |
| Time Period | 30 days (744 hours) |
| Fraud Cases | ~8,213 (0.13%) |
| Transaction Types | CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER |

> Fraud only exists in TRANSFER and CASH_OUT transaction types.

---

## 🧠 Model Architecture

Two specialist XGBoost models — one per transaction type:

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| TRANSFER Specialist | 100% | 98% | 0.99 | 1.0000 |
| CASH_OUT Specialist | 99% | 68% | 0.81 | 0.9985 |

> A single global model achieved AUC 0.9923. Segmentation pushed
> TRANSFER to perfect AUC 1.00 and reduced false alarms by 99%.

---

## ⚙️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| ML Models | XGBoost, Random Forest |
| Data Processing | Pandas, NumPy |
| Imbalance Handling | SMOTE |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn |
| Model Serving | Joblib |
| Interface | Streamlit |
| Environment | Google Colab |
| Version Control | GitHub |

---

## 🔄 ML Pipeline

Raw Data → Preprocessing → EDA → Feature Engineering → Model Training → Error Analysis → Segmentation → Streamlit App

### Key Engineering Decisions

- **Filtered to TRANSFER & CASH_OUT only** — fraud doesn't exist in other types
- **SMOTE** — handled 0.13% class imbalance without data leakage
- **Threshold tuning** — reduced false alarms from 29,634 to 291
- **Model segmentation** — specialist models outperform a single global model
- **SHAP** — every fraud flag comes with a human-readable explanation
- **Kept outliers** — extreme transactions are the fraud signal, not noise

---

## 🖥️ How to Use the App

1. Go to the live app link above
2. Upload a CSV file with transaction data
3. View the fraud summary dashboard
4. Filter by Fraud Only / Legit Only
5. Select any flagged transaction to see SHAP explanation
6. Download flagged transactions as CSV

### Expected CSV Format

Your CSV must contain these columns:

step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest

---

## 📁 Repository Structure

fraud-detection-machine-learning/
│
├── main.py                 # Streamlit app
├── prediction_helper.py    # Model loading, preprocessing, prediction, SHAP
├── transfer_model.pkl      # TRANSFER specialist model
├── cashout_model.pkl       # CASH_OUT specialist model
├── requirements.txt        # Python dependencies
└── README.md               # This file

---

## 🏗️ Run Locally

```bash
git clone https://github.com/productankur/fraud-detection-machine-learning.git
cd fraud-detection-machine-learning
pip install -r requirements.txt
streamlit run main.py
```

---

## 👤 Built By

**Ankur Ratwaya** — Product Manager × AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/ankurratwaya/)
