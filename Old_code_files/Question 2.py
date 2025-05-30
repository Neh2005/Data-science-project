# Question 2

import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd

# Load & filter
national_only = df1[df1['Level'] == "National"]
national_only = national_only[['Date', 'Number of Trips 10-25', 'Number of Trips 50-100']]
national_only['Date'] = dd.to_datetime(national_only['Date'], errors='coerce')

# Filter Dask datasets for > 10 million
set_10_25 = national_only[national_only['Number of Trips 10-25'] > 10_000_000]
set_50_100 = national_only[national_only['Number of Trips 50-100'] > 10_000_000]

# Convert to Pandas
set_10_25_pd = set_10_25.compute()
set_50_100_pd = set_50_100.compute()

# Clean up Date format
set_10_25_pd['Date'] = pd.to_datetime(set_10_25_pd['Date']).dt.date
set_50_100_pd['Date'] = pd.to_datetime(set_50_100_pd['Date']).dt.date

# Create sets of dates
dates_10_25 = set(set_10_25_pd['Date'])
dates_50_100 = set(set_50_100_pd['Date'])

common_dates = dates_10_25 & dates_50_100
only_10_25 = dates_10_25 - dates_50_100
only_50_100 = dates_50_100 - dates_10_25

# Create dataframes for scatter
only_10_25_df = set_10_25_pd[set_10_25_pd['Date'].isin(only_10_25)]
only_50_100_df = set_50_100_pd[set_50_100_pd['Date'].isin(only_50_100)]
both_df = set_10_25_pd[set_10_25_pd['Date'].isin(common_dates)]

# --- Print summary

print(f"\nTotal dates with >10M 10-25 trips: {len(dates_10_25)}")
print(f"Total dates with >10M 50-100 trips: {len(dates_50_100)}")
print(f"Common dates where both had >10M trips: {len(common_dates)}")


# Scatter: Trips 10–25 Miles > 10M
plt.figure(figsize=(10, 5))
plt.scatter(set_10_25_pd['Date'], set_10_25_pd['Number of Trips 10-25'],
            color='steelblue', alpha=0.6)
plt.title('Trips (10–25 Miles) > 10 Million')
plt.xlabel('Date')
plt.ylabel('Number of Trips 10–25')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter: Trips 50–100 Miles > 10M
plt.figure(figsize=(10, 5))
plt.scatter(set_50_100_pd['Date'], set_50_100_pd['Number of Trips 50-100'],
            color='darkorange', alpha=0.6)
plt.title('Trips (50–100 Miles) > 10 Million')
plt.xlabel('Date')
plt.ylabel('Number of Trips 50–100')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter: Both 10–25 and 50–100 Miles > 10M
plt.figure(figsize=(10, 5))
plt.scatter(both_df['Date'], both_df['Number of Trips 10-25'], 
            color='teal', label='10–25 Miles > 10M', alpha=0.6)
plt.scatter(both_df['Date'], both_df['Number of Trips 50-100'],
            color='crimson', label='50–100 Miles > 10M', alpha=0.6)
plt.title('Dates Where Both 10–25 and 50–100 Mile Trips > 10M')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
