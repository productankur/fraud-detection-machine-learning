
import pandas as pd
import numpy as np
import joblib
import shap

# ── Load models from repo root (not Google Drive) ─────────────────
transfer_model = joblib.load('transfer_model.pkl')
cashout_model  = joblib.load('cashout_model.pkl')

def preprocess(df):
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

    Q1    = df['amount'].quantile(0.25)
    Q3    = df['amount'].quantile(0.75)
    upper = Q3 + 1.5 * (Q3 - Q1)

    df['is_high_amount']        = (df['amount'] > upper).astype(int)
    df['sender_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_mismatch']      = (
        round(df['oldbalanceOrg'] - df['amount'], 2) !=
        round(df['newbalanceOrig'], 2)
    ).astype(int)
    df['type_encoded'] = (df['type'] == 'TRANSFER').astype(int)

    return df

def get_features(df):
    return df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
               'is_high_amount', 'balance_mismatch',
               'sender_balance_change', 'type_encoded']]

def predict(df):
    df            = preprocess(df)
    probabilities = []
    predictions   = []

    for idx, row in df.iterrows():
        features = get_features(df.loc[[idx]])
        model    = transfer_model if row['type_encoded'] == 1 else cashout_model
        prob     = model.predict_proba(features)[0][1]
        pred     = 1 if prob >= 0.5 else 0
        probabilities.append(round(prob, 4))
        predictions.append(pred)

    df['fraud_probability'] = probabilities
    df['prediction']        = predictions
    df['status']            = df['prediction'].map({1: '🔴 Fraud', 0: '🟢 Legit'})

    return df

def get_shap_values(df, transaction_type):
    model     = transfer_model if transaction_type == 'TRANSFER' else cashout_model
    features  = get_features(df)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(features)
    return shap_vals, features
