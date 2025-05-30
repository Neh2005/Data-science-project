import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import time
from dask.distributed import Client, LocalCluster
import os

# Load data
df1_pandas = pd.read_csv("trips_by_distance.csv.csv", low_memory=False)

# Check your machine
physical_cores = os.cpu_count()
print(f"Your machine has {physical_cores} logical cores")

# Test configurations simulating 10 and 20 processors (via partition count)
test_configs = {
    10: 10,
    20: 20,
}

processor_times = {}
results = {}

# Run for each simulated processor config
for label, partitions in test_configs.items():
    print(f"\n Simulating {label} processors using {partitions} partitions...")

    cluster = LocalCluster(n_workers=physical_cores, threads_per_worker=1)
    client = Client(cluster)

    start = time.time()

    ddf = dd.from_pandas(df1_pandas, npartitions=partitions)
    ddf['Date'] = dd.to_datetime(ddf['Date'], errors='coerce')

    # Filter for 'National' level and >10M trip values (real task!)
    ddf_filtered = ddf[ddf['Level'] == 'National']
    ddf_filtered = ddf_filtered[['Date', 'Number of Trips 10-25', 'Number of Trips 50-100']]
    filtered_result = ddf_filtered[
        (ddf_filtered['Number of Trips 10-25'] > 10_000_000) |
        (ddf_filtered['Number of Trips 50-100'] > 10_000_000)
    ].compute()

    end = time.time()
    processor_times[label] = end - start
    results[label] = filtered_result

    print(f" Time using {label} processors: {end - start:.4f} seconds")

    client.close()
    cluster.close()

# Sequential processing (baseline)
print("\n Running baseline (Pandas, sequential)...")
start_serial = time.time()

df1_pandas['Date'] = pd.to_datetime(df1_pandas['Date'], errors='coerce')
df1_clean = df1_pandas[df1_pandas['Level'] == 'National']
df1_clean = df1_clean[['Date', 'Number of Trips 10-25', 'Number of Trips 50-100']]
df1_filtered_seq = df1_clean[
    (df1_clean['Number of Trips 10-25'] > 10_000_000) |
    (df1_clean['Number of Trips 50-100'] > 10_000_000)
]

end_serial = time.time()
sequential_time = end_serial - start_serial
print(f" Sequential time: {sequential_time:.4f} seconds")

# Add to comparison dict
processor_times['Sequential'] = sequential_time

# Plotting
# Fix: Use numeric x-positions instead of string labels
labels = list(processor_times.keys())
times = list(processor_times.values())
x_pos = list(range(len(labels)))

plt.figure(figsize=(9, 5))
plt.bar(x_pos, times, color=['lightcoral' if l == 'Sequential' else 'skyblue' for l in labels])
plt.xticks(x_pos, labels)  # Assign string labels to x-ticks
plt.xlabel("Processor Simulation")
plt.ylabel("Processing Time (seconds)")
plt.title("Q2:Parallel vs Sequential Processing Time")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

