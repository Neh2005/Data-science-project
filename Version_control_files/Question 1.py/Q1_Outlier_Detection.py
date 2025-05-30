#Just a box plot visualizationbaseed on the national level data in both the datasets

import matplotlib.pyplot as plt

# Convert Dask DataFrames to pandas
df1_pd = df1.compute()
df2_pd = df2.compute()
# Convert Dask DataFrames to pandas
df1_national = df1_pd[df1_pd['Level'] == 'National']
df2_national = df2_pd[df2_pd['Level'] == 'National']

# First dataset outlier graph based on National level
plt.figure(figsize=(8, 6))
plt.boxplot([
    df1_national['Population Staying at Home'].dropna(),
    df1_national['Population Not Staying at Home'].dropna()
], labels=['Staying (df1)', 'Not Staying (df1)'])
plt.title('Outliers in Population Staying vs Not Staying at Home (df1)')
plt.ylabel('Population')
plt.grid(True)
plt.show()


# Second dataset outlier graph based on National level
plt.figure(figsize=(8, 6))
plt.boxplot([
    df2_national['Population Staying at Home'].dropna(),
    df2_national['People Not Staying at Home'].dropna()
], labels=['Staying (df2)', 'Not Staying (df2)'])
plt.title('Outliers in Population Staying vs Not Staying at Home (df2)')
plt.ylabel('Population')
plt.grid(True)
plt.show()

