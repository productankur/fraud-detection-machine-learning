
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction_helper import predict, get_shap_values

st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="wide")
st.title("🔍 Fraud Detection System")
st.markdown("Upload a transaction CSV file to detect fraudulent transactions.")
st.divider()

uploaded_file = st.file_uploader(
    "Upload Transaction CSV", type=['csv'],
    help="CSV must contain: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.spinner("Running fraud detection..."):
        results = predict(df)

    st.subheader("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    total       = len(results)
    fraud_count = results['prediction'].sum()
    legit_count = total - fraud_count
    fraud_rate  = (fraud_count / total * 100) if total > 0 else 0

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("🔴 Fraud Detected",  f"{fraud_count:,}")
    col3.metric("🟢 Legit",           f"{legit_count:,}")
    col4.metric("Fraud Rate",         f"{fraud_rate:.2f}%")

    st.divider()

    st.subheader("📋 Transaction Results")
    filter_option = st.radio("Filter transactions:",
                             ["All", "Fraud Only", "Legit Only"], horizontal=True)

    display_cols = ['nameOrig', 'type', 'amount',
                    'sender_balance_change', 'fraud_probability', 'status']

    if filter_option == "Fraud Only":
        display_df = results[results['prediction'] == 1][display_cols]
    elif filter_option == "Legit Only":
        display_df = results[results['prediction'] == 0][display_cols]
    else:
        display_df = results[display_cols]

    st.dataframe(
        display_df.style.apply(
            lambda x: ['background-color: #ffcccc'
                       if v == '🔴 Fraud' else '' for v in x],
            subset=['status']
        ),
        use_container_width=True
    )

    fraud_only = results[results['prediction'] == 1]
    st.download_button(
        label     = "⬇️ Download Fraud Transactions CSV",
        data      = fraud_only.to_csv(index=False),
        file_name = "fraud_transactions.csv",
        mime      = "text/csv"
    )

    st.divider()

    st.subheader("🧠 Why Were These Flagged?")
    fraud_rows = results[results['prediction'] == 1]

    if fraud_rows.empty:
        st.success("No fraud detected in this batch.")
    else:
        selected = st.selectbox("Select a flagged transaction", fraud_rows['nameOrig'].values)
        row      = fraud_rows[fraud_rows['nameOrig'] == selected]
        txn_type = row['type'].values[0]
        amount   = row['amount'].values[0]
        prob     = row['fraud_probability'].values[0]

        shap_vals, features = get_shap_values(row, txn_type)

        c1, c2, c3 = st.columns(3)
        c1.metric("Type",              txn_type)
        c2.metric("Amount",            f"₹{amount:,.0f}")
        c3.metric("Fraud Probability", f"{prob:.0%}")

        st.divider()

        feature_names = features.columns.tolist()
        shap_values   = shap_vals[0]
        sorted_idx    = np.argsort(np.abs(shap_values))
        display_vals  = shap_values[sorted_idx]
        display_names = [feature_names[i] for i in sorted_idx]
        colors        = ['#E24B4A' if v > 0 else '#378ADD' for v in display_vals]

        n = len(display_names)
        fig, ax = plt.subplots(figsize=(7, max(3, n * 0.55)))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        bars = ax.barh(display_names, display_vals,
                       color=colors, height=0.55, edgecolor='none')

        for bar, val in zip(bars, display_vals):
            ax.text(
                val + (0.008 if val >= 0 else -0.008),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=9, color='white'
            )

        ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        lc1, lc2 = st.columns(2)
        lc1.markdown("🔴 Pushed towards **fraud**")
        lc2.markdown("🔵 Pushed towards **legit**")
