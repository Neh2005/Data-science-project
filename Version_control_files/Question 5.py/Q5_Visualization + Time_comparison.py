import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time
from dask.distributed import Client, LocalCluster
import os

# File paths
file1 = "trips_by_distance.csv.csv"
file2 = "trips_full_data__2_.csv"

# Load df2 (smaller one) as Pandas always
df2_pandas = pd.read_csv(file2, low_memory=False)
df2_pandas['Date'] = pd.to_datetime(df2_pandas['Date'], errors='coerce')

# Determine common date range
start_date = df2_pandas['Date'].min()
end_date = df2_pandas['Date'].max()

# Timing results
viz_times = {}

# Test with parallel configs
test_configs = {
    "Parallel-10": 10,
    "Parallel-20": 20
}

for label, nparts in test_configs.items():
    print(f"\n Running {label} with {nparts} partitions...")

    cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1)
    client = Client(cluster)

    start = time.time()

    # Load df1 in Dask
    ddf1 = dd.read_csv(file1, assume_missing=True, dtype=str)
    ddf1['Date'] = dd.to_datetime(ddf1['Date'], errors='coerce')

    # Filter National-level & date range
    ddf1 = ddf1[ddf1['Level'] == "National"]
    ddf1 = ddf1[ddf1["Date"].between(start_date, end_date)]
    ddf1 = ddf1.drop(columns=['State FIPS', 'State Postal Code', 'County FIPS', 'County Name'], errors='ignore')

    # df2 cleanup
    df2_clean = df2_pandas.drop(
        columns=["Month of Date", "Week of Date", "Year of Date", "Level", "Date", "Trips", "Week Ending Date"],
        errors='ignore'
    )

    # Calculate means using Dask
    ddf2 = dd.from_pandas(df2_clean, npartitions=nparts)
    trip_means = ddf2.select_dtypes(include='number').mean().compute()

    end = time.time()
    viz_times[label] = end - start

    print(f" {label} completed in {end - start:.4f} seconds")

    client.close()
    cluster.close()

# === Serial (Pandas) Version ===
print("\n Running Serial (Pandas) Version...")
start_serial = time.time()

# Load as Pandas
df1 = pd.read_csv(file1, low_memory=False)
df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')

# Filter
df1 = df1[df1['Level'] == "National"]
df1 = df1[df1["Date"].between(start_date, end_date)]
df1 = df1.drop(columns=['State FIPS', 'State Postal Code', 'County FIPS', 'County Name'], errors='ignore')

# Clean df2 again
df2_clean = df2_pandas.drop(
    columns=["Month of Date", "Week of Date", "Year of Date", "Level", "Date", "Trips", "Week Ending Date"],
    errors='ignore'
)

trip_means_serial = df2_clean.select_dtypes(include='number').mean()

end_serial = time.time()
viz_times['Sequential'] = end_serial - start_serial
print(f" Serial version completed in {end_serial - start_serial:.4f} seconds")

# === Plot bar chart for one version (trip_means_serial or latest parallel) ===
plt.figure(figsize=(12, 6))
trip_means_serial.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Average Number of Trips by Distance (National Level)')
plt.xlabel('Distance Categories')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# === Compare Timing ===
labels = list(viz_times.keys())
times = list(viz_times.values())
x_pos = list(range(len(labels)))

plt.figure(figsize=(9, 5))
plt.bar(x_pos, times, color=['lightcoral' if 'Sequential' in l else 'skyblue' for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Processing Time (seconds)")
plt.title("Sequential vs Parallel Processing Time – Trip Distance Averages")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Summary
print("\n Summary:")
for label in labels:
    print(f"{label}: {viz_times[label]:.4f} seconds")
