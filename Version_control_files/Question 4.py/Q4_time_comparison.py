import pandas as pd
import numpy as np
import time
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# File paths
file1 = "trips_by_distance.csv.csv"
file2 = "trips_full_data__2_.csv"

# For storing results
model_times = {}

# Simulation configs
configs = {
    10: 10,   # simulate 10 processors
    20: 20    # simulate 20 processors
}

# Load df2 as pandas (since it's smaller and less filtered)
df2_pandas = pd.read_csv(file2, low_memory=False)
df2_pandas['Date'] = pd.to_datetime(df2_pandas['Date'], errors='coerce')

# Dates
start_date = df2_pandas['Date'].min()
end_date = df2_pandas['Date'].max()

for label, nparts in configs.items():
    print(f"\n Parallel Simulation: {label} processors, {nparts} partitions")

    # Setup Dask cluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    start = time.time()

    # Load df1 via Dask
    ddf = dd.read_csv(file1,assume_missing=True,dtype=str,  # Avoids dtype inference bugs
    blocksize="64MB"  # Smaller chunks, avoids RAM overload
)

    ddf['Date'] = dd.to_datetime(ddf['Date'], errors='coerce')
    ddf = ddf[ddf['Level'] == 'National']
    ddf = ddf[ddf['Date'].between(start_date, end_date)]
    ddf = ddf[['Date', 'Number of Trips 5-10']]

    df1_filtered = ddf.compute()

    # Preprocess df2
    df2_filtered = df2_pandas[df2_pandas['Date'].between(start_date, end_date)][['Date', 'Trips 25-50 Miles']]

    # Align & merge
    merged = pd.merge(df1_filtered, df2_filtered, on='Date', how='inner').dropna()
    X = merged[['Trips 25-50 Miles']]
    y = merged[['Number of Trips 5-10']]

    # Model training
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Evaluate
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    end = time.time()
    model_times[f"Parallel-{label}"] = end - start

    print(f" R²: {r2:.4f}, Time: {end - start:.4f}s")

    client.close()
    cluster.close()

# === Serial version ===
print("\n Serial (Pandas) Version")

start_serial = time.time()

# Load with Pandas
df1 = pd.read_csv(file1, low_memory=False)
df2 = pd.read_csv(file2, low_memory=False)

# Clean & filter
df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

df1 = df1[df1['Level'] == "National"]
df1 = df1[df1['Date'].between(start_date, end_date)]
df2 = df2[df2['Date'].between(start_date, end_date)]

# Merge & train
merged = pd.merge(df1[['Date', 'Number of Trips 5-10']],
                  df2[['Date', 'Trips 25-50 Miles']],
                  on='Date').dropna()

X = merged[['Trips 25-50 Miles']]
y = merged[['Number of Trips 5-10']]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

end_serial = time.time()
model_times['Sequential'] = end_serial - start_serial

print(f" R²: {r2:.4f}, Time: {end_serial - start_serial:.4f}s")

# === Plot Comparison ===
import matplotlib.pyplot as plt

labels = list(model_times.keys())
times = list(model_times.values())
x_pos = list(range(len(labels)))

plt.figure(figsize=(9, 5))
plt.bar(x_pos, times, color=['skyblue' if "Parallel" in l else 'lightcoral' for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Time (seconds)")
plt.title("Model Training & Evaluation: Sequential vs Simulated Parallel")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
