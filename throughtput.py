import time
import random
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# ======================================
# LOAD TRAINED MODEL
# ======================================

print("Loading model...")

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model loaded successfully!")

# ======================================
# SIMULATION SETTINGS
# ======================================

TOTAL_TRANSACTIONS = 10000   # change to test larger loads

print("\nStarting throughput test...")
print("Transactions to simulate:", TOTAL_TRANSACTIONS)

# ======================================
# STORAGE FOR LATENCY
# ======================================

latencies = []

# ======================================
# START TEST
# ======================================

start_time = time.time()

for i in range(TOTAL_TRANSACTIONS):

    # ----------------------------
    # Simulate transaction features
    # ----------------------------

    debit_amt = random.randint(100, 50000)
    hour = random.randint(0, 23)
    time_gap_sec = random.randint(1, 3600)
    velocity_1h = random.randint(1, 5)

    balance_drain_pct = random.random()

    ip_change_flag = random.randint(0, 1)
    location_change_flag = random.randint(0, 1)

    amount_spike_flag = random.randint(0, 1)

    features = [[
        debit_amt,
        hour,
        time_gap_sec,
        velocity_1h,
        balance_drain_pct,
        ip_change_flag,
        location_change_flag,
        amount_spike_flag
    ]]

    # ----------------------------
    # Measure latency
    # ----------------------------

    tx_start = time.time()
    col_names = [
        'Debited Amt', 'hour', 'time_gap_sec', 'velocity_1h',
        'balance_drain_pct', 'ip_change_flag', 'location_change_flag',
        'Amount_Spike_Flag'
    ]
    input_df = pd.DataFrame(features, columns=col_names)
    scaled_features = scaler.transform(input_df)

    score = model.decision_function(scaled_features)

    tx_end = time.time()

    latency = tx_end - tx_start
    latencies.append(latency)

# ======================================
# END TEST
# ======================================

end_time = time.time()

total_time = end_time - start_time

throughput = TOTAL_TRANSACTIONS / total_time

avg_latency = np.mean(latencies)

max_latency = np.max(latencies)

# ======================================
# RESULTS
# ======================================

print("\n===================================")
print("THROUGHPUT SCALABILITY RESULTS")
print("===================================")

print("Total Transactions Processed:", TOTAL_TRANSACTIONS)

print("Total Time Taken:", round(total_time, 3), "seconds")

print("Throughput (Transactions Per Second):", round(throughput, 2), "TPS")

print("Average Latency Per Transaction:", round(avg_latency * 1000, 3), "ms")

print("Maximum Latency:", round(max_latency * 1000, 3), "ms")

print("===================================")