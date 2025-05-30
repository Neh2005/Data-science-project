import numpy as np
import pandas as pd
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt 


# For reading both the datasets
file = "trips_by_distance.csv.csv"
file2 =  "trips_full_data__2_.csv"


# To convert the datatypes of required columns of large dataset
dtypes = {
    "County Name": "object",
    "State Postal Code": "object",
    "Level": "object",
    "Date": "object",
    "State FIPS": "object",
    "County FIPS": "float64",
    "Population Staying at Home": "float64",
    "Population Not Staying at Home": "float64",
    "Number of Trips": "float64",
    "Number of Trips <1": "float64",
    "Number of Trips 1-3": "float64",
    "Number of Trips 3-5": "float64",
    "Number of Trips 5-10": "float64",
    "Number of Trips 10-25": "float64",
    "Number of Trips 25-50": "float64",
    "Number of Trips 50-100": "float64",
    "Number of Trips 100-250": "float64",
    "Number of Trips 250-500": "float64",
    "Number of Trips >=500": "float64"
}

# Since trips_by_distance is a large dataset with almost 100000 values, dask is best to handle it.
df1 = dd.read_csv(file, dtype = dtypes, blocksize=None)
df2 = dd.read_csv(file2)


print("First large dataset is loaded successfully after conversions!")
print(df1.dtypes)


print(df1.isna().sum().compute())
print(" ")
print(df2.isna().sum().compute())
print(" ")
print("First dataset:", df1.info())
print(" ")
print("Second Dataset:",df2.info())



# Since only the large dataset has null values use fillna() for df1 and drop duplicates.
df1 = df1.fillna(0)
df1 = df1.drop_duplicates()

#Again check for nullvalues
print(df1.isna().sum().compute())

# Since date is an important part, converting the dates in both th edatasets to the uniform datetime format
df1['Date'] = dd.to_datetime(df1['Date'])  
df2['Date'] = dd.to_datetime(df2['Date'])

# With this data cleaning steps are finished.
