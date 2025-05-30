# Benchmarking .describe() performance using Dask with varying partition counts (simulated 10 and 20 processors).
# Compared Dask execution time against Pandas (sequential) to evaluate parallel efficiency.

import pandas as pd
import dask.dataframe as dd
import time
from dask.distributed import Client, LocalCluster
import os

# Load CSV via Pandas
df1_pandas = pd.read_csv("trips_by_distance.csv.csv", low_memory=False)

# CPU info
physical_cores = os.cpu_count()  # should be 8
print(f"Your machine has {physical_cores} logical cores")

# Worker configs: Fixed workers, varying partitions (simulating more "parallelism")
test_configs = {
    10: 32,  # 10 partitions
    20: 20,  # 20 partitions
}

processor_times = {}

for label, partitions in test_configs.items():
    print(f"\n🔧 Simulating {label} processors with {physical_cores} actual workers and {partitions} partitions")

    # Set cluster with 8 actual workers (your CPU limit)
    cluster = LocalCluster(n_workers=physical_cores, threads_per_worker=1)
    client = Client(cluster)

    start = time.time()

    # Dask DataFrame with simulated partition count
    ddf = dd.from_pandas(df1_pandas, npartitions=partitions)

    # Some meaningful operation
    ddf['Date'] = dd.to_datetime(ddf['Date'], errors='coerce')
    result = ddf.describe().compute()

    end = time.time()
    processor_times[label] = end - start

    print(f" Time simulating {label} processors: {end - start:.4f} seconds")

    client.close()
    cluster.close()
    start_serial = time.time()
    operate = df1_pandas.describe()
    end_serial = time.time()
    print(f"Serial (Pandas) processing time: {end_serial - start_serial:.4f} seconds")


# Summary
print("\n Processing Time Comparison:")
for simulated, runtime in processor_times.items():
    print(f"Simulated {simulated} processors: {runtime:.4f} seconds")
    
labels = list(processor_times.keys())
times = list(processor_times.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, times, color='skyblue')
plt.xlabel("Simulated Number of Processors (via Partitions)")
plt.ylabel("Processing Time (seconds)")
plt.title("Processing Time vs Simulated Number of Processors")
plt.xticks(labels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
