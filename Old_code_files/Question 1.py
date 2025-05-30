##Question 1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Filter only national-level data from first large dataset
national_level = df1[df1['Level'] == "National"]

# 2. Drop unnecessary columns from first large dataset
national_level = national_level.drop(columns=[
    'State FIPS', 'State Postal Code', 'County FIPS', 'County Name'
], errors='ignore')

# Remove extra spaces from second dataset
df2.columns = df2.columns.str.strip()  

#Drop unnecessary columns in second small dataset
df2 = df2.drop(columns=[
    "Month of Date", "Week of Date", "Year of Date", "Level",
    "Trips", "Week Ending Date"
], errors='ignore')

# 3. Filter both datasets to matching date ranges
national_level = national_level[
    national_level["Date"].between(df2['Date'].min(), df2['Date'].max())
].compute()

df2 = df2[
    df2["Date"].between(national_level['Date'].min(), national_level['Date'].max())
]

df2=df2.compute()

# 4. Group national data by date (only numeric columns) to find the mean 
numeric_national = national_level.select_dtypes(include='number')
grouped_national = national_level.groupby('Date')[numeric_national.columns].mean()

numeric_df2 = df2.select_dtypes(include='number')
grouped_small = df2.groupby('Date')[numeric_df2.columns].mean()


# 5. Plot: Population Staying at Home + Trendline
plt.figure(figsize=(10, 5))
x = grouped_national.index.map(pd.Timestamp.toordinal)
y = grouped_national['Population Staying at Home']
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.bar(grouped_national.index, y, color='steelblue', label='Staying at Home')
plt.plot(grouped_national.index, p(x), color='orange', linewidth=2.5, linestyle='--', label='Trendline')
plt.xlabel('Date')
plt.ylabel('Population')
plt.title('Population Staying at Home (National Level)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# 6. Plot: Population Not Staying at Home + Trendline
plt.figure(figsize=(10, 5))
x2 = grouped_small.index.map(pd.Timestamp.toordinal)
y2 = grouped_small['People Not Staying at Home']
z2 = np.polyfit(x2, y2, 1)
p2 = np.poly1d(z2)
plt.bar(grouped_small.index, y2, color='tomato', label='Not Staying at Home')
plt.plot(grouped_small.index, p2(x2), color='black', linewidth=2.5,linestyle='--', label='Trendline')
plt.xlabel('Date')
plt.ylabel('Population')
plt.title('Population Not Staying at Home (National Level)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

    
  
