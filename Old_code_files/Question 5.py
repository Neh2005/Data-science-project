import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd

# 1. Filter only National-level data from the big dataset
national_only = df1[df1['Level'] == "National"]

# 2. Drop location-related columns (they're all NaN for National)
national_only = national_only.drop(
    columns=['State FIPS', 'State Postal Code', 'County FIPS', 'County Name'],
    errors='ignore'
)

# 3. Filter df1 by the date range in df2
start_date = df2['Date'].min().compute() if isinstance(df2, dd.DataFrame) else df2['Date'].min()
end_date = df2['Date'].max().compute() if isinstance(df2, dd.DataFrame) else df2['Date'].max()
national_only = national_only[national_only["Date"].between(start_date, end_date)]

# 4. Clean df2 by dropping unnecessary columns
df2 = df2.drop(
    columns=["Month of Date", "Week of Date", "Year of Date", "Level", "Date", "Trips", "Week Ending Date"],
    errors='ignore')

# 5. Calculate mean number of trips per distance
trip_means = df2.select_dtypes(include='number').mean().compute() if isinstance(df2, dd.DataFrame) else df2.select_dtypes(include='number').mean()

# 6. Plot Distance vs Average Trips
plt.figure(figsize=(12, 6))
trip_means.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Average Number of Trips by Distance (National Level)')
plt.xlabel('Distance Categories')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


