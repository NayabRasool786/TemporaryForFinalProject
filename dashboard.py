import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

iso_model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Fraud Detection Monitor",
    layout="wide"
)

st.title("🏦 Real-Time Fraud Detection Dashboard")

# ------------------------------
# LOAD DATA
# ------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("processed_transactions.csv")

df = load_data()

# ------------------------------
# KPIs
# ------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(df))
col2.metric("Total Fraud", df['FRAUD'].sum())
col3.metric("Fraud Rate", round(df['FRAUD'].mean()*100,3))
col4.metric("Avg Risk Score", round(df['final_risk'].mean(),3))

st.divider()

# ------------------------------
# FRAUD VS NORMAL
# ------------------------------

st.subheader("Fraud vs Normal Transactions")

fraud_counts = df['FRAUD'].value_counts().reset_index()
fraud_counts.columns = ["Type","Count"]

fig = px.pie(
    fraud_counts,
    values="Count",
    names="Type",
    color="Type",
    color_discrete_map={0:"green",1:"red"}
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# RISK SCORE DISTRIBUTION
# ------------------------------

st.subheader("Risk Score Distribution")

fig = px.histogram(
    df,
    x="final_risk",
    nbins=50,
    title="Risk Score Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# FRAUD LOCATIONS
# ------------------------------

st.subheader("Fraud Location Analysis")

fraud_locations = df[df["FRAUD"]==1]["Location"].value_counts()

fig = px.bar(
    fraud_locations,
    title="Fraud Transactions by Location"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# RISK BAND ANALYSIS
# ------------------------------

def risk_band(score):

    if score < 0.4:
        return "Low"
    elif score < 0.65:
        return "Medium"
    elif score < 0.8:
        return "High"
    else:
        return "Critical"

df["risk_band"] = df["final_risk"].apply(risk_band)

band_data = df.groupby(["risk_band","FRAUD"]).size().reset_index(name="count")

fig = px.bar(
    band_data,
    x="risk_band",
    y="count",
    color="FRAUD",
    barmode="group",
    title="Fraud Distribution by Risk Band"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# TRANSACTION TABLE
# ------------------------------

st.subheader("Transaction Monitor")

st.dataframe(
    df[['TransactionID','Debited Amt','Location','final_risk','decision']],
    use_container_width=True
)









import numpy as np
import pandas as pd

st.divider()
st.header("🔎 Test a New Transaction")

# -------- INPUT FIELDS --------
device_ip = st.text_input("Device IP", "49.207.45.128")

location = st.selectbox(
    "Location",
    df["Location"].dropna().unique()
)

transaction_time = st.text_input(
    "Transaction Time (YYYY-MM-DD HH:MM:SS)",
    "2021-02-16 14:20:00"
)

debit_amt = st.number_input(
    "Debit Amount",
    min_value=0.0,
    value=2000.0
)

# -------- BUTTON --------
if st.button("Check Transaction"):

    # convert time
    transaction_time = pd.to_datetime(transaction_time)

    hour = transaction_time.hour
    is_night = int(0 <= hour <= 5)

    last_row = df.iloc[-1]

    prev_balance = last_row["Total Amt"]

    balance_drain_pct = debit_amt / prev_balance

    ip_change_flag = int(device_ip != last_row["Device IP"])

    location_change_flag = int(location != last_row["Location"])

    time_gap_sec = (
        transaction_time -
        pd.to_datetime(last_row["Transaction Time"])
    ).total_seconds()

    if time_gap_sec < 0:
        time_gap_sec = 0

    velocity_1h = 1

    # -------- Z-SCORES --------
    debit_zscore = (
        debit_amt - df["Debited Amt"].mean()
    ) / df["Debited Amt"].std()

    velocity_zscore = (
        velocity_1h - df["velocity_1h"].mean()
    ) / df["velocity_1h"].std()

    # -------- ANOMALY MODEL --------
    anomaly_input = scaler.transform([[

        debit_amt,
        hour,
        time_gap_sec,
        velocity_1h,
        balance_drain_pct,
        ip_change_flag,
        location_change_flag,
        0

    ]])

    anomaly_score = -iso_model.decision_function(anomaly_input)[0]

    anomaly_score_norm = (
        anomaly_score - df["anomaly_score"].min()
    ) / (
        df["anomaly_score"].max() -
        df["anomaly_score"].min()
    )

    # -------- RULE ENGINE --------
    rule_score = 0

    if debit_zscore > 3:
        rule_score += 40

    if balance_drain_pct > 0.5:
        rule_score += 50

    if is_night:
        rule_score += 10

    if ip_change_flag:
        rule_score += 20

    if location_change_flag:
        rule_score += 20

    if velocity_zscore > 3:
        rule_score += 30

    rule_score_norm = rule_score / df["rule_score"].max()

    # -------- FINAL RISK --------
    final_risk = (
        0.6 * anomaly_score_norm +
        0.4 * rule_score_norm
    )

    # -------- DECISION --------
    if final_risk < 0.3:
        decision = "ALLOW"
    elif final_risk < 0.7:
        decision = "OTP_REQUIRED"
    else:
        decision = "BLOCK"

    # -------- OUTPUT --------
    st.subheader("Result")

    st.write("Risk Score:", round(final_risk,3))
    st.write("Decision:", decision)