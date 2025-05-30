# Benchmarking .describe() performance using Dask with varying partition counts (simulated 10 and 20 processors).
# Compared Dask execution time against Pandas (sequential) to evaluate parallel efficiency.
#Improved the code by including repeated trials, which reduces noise from background processes.

import pandas as pd
import dask.dataframe as dd
import time
from dask.distributed import Client, LocalCluster
import os
import matplotlib.pyplot as plt
import numpy as np

# Load CSV with Pandas
df = pd.read_csv("trips_by_distance.csv.csv", low_memory=False)

# Use only numeric columns for performance test
df_numeric = df.select_dtypes(include='number')

# CPU info
physical_cores = os.cpu_count()
print(f"Your machine has {physical_cores} logical cores")

# Simulate processor performance by changing number of partitions
configs = {
    10: 10,
    20: 20,
}

processor_times = {}

for processors, partitions in configs.items():
    print(f"\n Running with {processors} processors and {partitions} partitions")

    cluster = LocalCluster(n_workers=physical_cores, threads_per_worker=1)
    client = Client(cluster)

    times = []
    for trial in range(3):
        print(f"Trial {trial + 1}...")

        start = time.time()

        # Create Dask DataFrame
        ddf = dd.from_pandas(df_numeric, npartitions=partitions)

        # Lightweight compute
        result = ddf.mean().compute()

        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f" Time: {elapsed:.4f} sec")

    avg_time = np.mean(times)
    processor_times[processors] = avg_time
    print(f" Avg Time with {processors} processors: {avg_time:.4f} sec")

    client.close()
    cluster.close()
    # Sequential processing with Pandas (baseline)
print("\n Running sequential processing with Pandas...")

times_seq = []
for trial in range(3):
    print(f"Trial {trial + 1}...")
    start = time.time()

    result_seq = df_numeric.mean()  # Pandas computes eagerly

    end = time.time()
    elapsed = end - start
    times_seq.append(elapsed)
    print(f" Time: {elapsed:.4f} sec")

avg_seq_time = np.mean(times_seq)
processor_times["Sequential"] = avg_seq_time
print(f" Avg Sequential Time: {avg_seq_time:.4f} sec")


# First Plot: Dask Performance Only (Exclude Sequential)
dask_only = {k: v for k, v in processor_times.items() if k != "Sequential"}

plt.figure(figsize=(8, 5))
plt.bar(list(dask_only.keys()), list(dask_only.values()), color='skyblue')
plt.xlabel("Number of Dask Partitions (Simulated Processors)")
plt.ylabel("Avg Processing Time (seconds)")
plt.title("Dask Performance: Partitions vs Processing Time")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Second Plot: Dask vs Pandas Sequential
plt.figure(figsize=(8, 5))

# Combine all configurations including Sequential
all_keys = list(dask_only.keys()) + ["Sequential"]
all_values = [processor_times[k] for k in all_keys]

# Map labels to numeric x-axis positions
x_pos = np.arange(len(all_keys))

plt.bar(x_pos, all_values, color='orange')
plt.xticks(x_pos, all_keys)  # Set string labels on x-axis
plt.xlabel("Execution Mode")
plt.ylabel("Avg Processing Time (seconds)")
plt.title("Dask vs Pandas Performance Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


