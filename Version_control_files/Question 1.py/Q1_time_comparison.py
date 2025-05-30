import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import time
from dask.distributed import Client, LocalCluster
import os

# Load datasets
file1 = "trips_by_distance.csv.csv"
file2 = "trips_full_data__2_.csv"

df1_pandas = pd.read_csv(file1, low_memory=False)
df2_pandas = pd.read_csv(file2, low_memory=False)
df2_pandas['Date'] = pd.to_datetime(df2_pandas['Date'], errors='coerce')

# Check system cores
physical_cores = os.cpu_count()
print(f"Your machine has {physical_cores} logical cores")

# Simulate parallel runs
test_configs = {
    10: 10,
    20: 20,
}

processor_times = {}

# Run for each parallel config
for label, partitions in test_configs.items():
    print(f"\n🔧 Simulating {label} processors using {partitions} partitions...")

    cluster = LocalCluster(n_workers=physical_cores, threads_per_worker=1)
    client = Client(cluster)

    start = time.time()

    ddf = dd.from_pandas(df1_pandas.copy(), npartitions=partitions)
    ddf['Date'] = dd.to_datetime(ddf['Date'], errors='coerce')

    df2_clean = df2_pandas.drop(columns=[
        "Month of Date", "Week of Date", "Year of Date", "Level",
        "Trips", "Week Ending Date"
    ], errors='ignore')
    df2_clean.columns = df2_clean.columns.str.strip()

    national_level = ddf[ddf['Level'] == "National"]
    national_level = national_level.drop(columns=[
        'State FIPS', 'State Postal Code', 'County FIPS', 'County Name'
    ], errors='ignore')

    national_level = national_level[
        national_level["Date"].between(df2_pandas['Date'].min(), df2_pandas['Date'].max())
    ].compute()

    df2_filtered = df2_clean[
        df2_clean["Date"].between(national_level['Date'].min(), national_level['Date'].max())
    ]

    numeric_national = national_level.select_dtypes(include='number')
    grouped_national = national_level.groupby('Date')[numeric_national.columns].mean()

    numeric_df2 = df2_filtered.select_dtypes(include='number')
    grouped_small = df2_filtered.groupby('Date')[numeric_df2.columns].mean()

    _ = grouped_national['Population Staying at Home'].mean()
    _ = grouped_small['People Not Staying at Home'].mean()

    end = time.time()
    processor_times[f"Parallel-{label}"] = end - start
    print(f"Time with {label} processors: {end - start:.4f} seconds")

    client.close()
    cluster.close()

# Sequential version (Pandas baseline)
print("\nRunning baseline (Sequential Pandas)...")

start_serial = time.time()

df1_pandas['Date'] = pd.to_datetime(df1_pandas['Date'], errors='coerce')
df2_pandas['Date'] = pd.to_datetime(df2_pandas['Date'], errors='coerce')

df2_clean = df2_pandas.drop(columns=[
    "Month of Date", "Week of Date", "Year of Date", "Level",
    "Trips", "Week Ending Date"
], errors='ignore')
df2_clean.columns = df2_clean.columns.str.strip()

national_level = df1_pandas[df1_pandas['Level'] == "National"]
national_level = national_level.drop(columns=[
    'State FIPS', 'State Postal Code', 'County FIPS', 'County Name'
], errors='ignore')

national_level = national_level[
    national_level["Date"].between(df2_pandas['Date'].min(), df2_pandas['Date'].max())
]

df2_filtered = df2_clean[
    df2_clean["Date"].between(national_level['Date'].min(), national_level['Date'].max())
]

numeric_national = national_level.select_dtypes(include='number')
grouped_national = national_level.groupby('Date')[numeric_national.columns].mean()

numeric_df2 = df2_filtered.select_dtypes(include='number')
grouped_small = df2_filtered.groupby('Date')[numeric_df2.columns].mean()

_ = grouped_national['Population Staying at Home'].mean()
_ = grouped_small['People Not Staying at Home'].mean()

end_serial = time.time()
processor_times['Sequential'] = end_serial - start_serial
print(f"Sequential time: {end_serial - start_serial:.4f} seconds")

# === Plotting
labels = list(processor_times.keys())
times = list(processor_times.values())
x_pos = list(range(len(labels)))

plt.figure(figsize=(9, 5))
plt.bar(x_pos, times, color=['lightcoral' if l == 'Sequential' else 'skyblue' for l in labels])
plt.xticks(x_pos, labels)
plt.xlabel("Processor Simulation")
plt.ylabel("Processing Time (seconds)")
plt.title("Q1: Sequential vs Parallel Processing Time")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Summary
print("\n Summary:")
for label in labels:
    print(f"{label}: {processor_times[label]:.4f} seconds")
