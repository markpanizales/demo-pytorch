# Identifying missing values in a tabular data
import pandas as pd
from io import StringIO
import sys

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0
'''
df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())

# Access the underlying numpy array
print(df.values)

# Drop missing values easily
print(df.dropna(axis=0))

# Drop one NaN in any row
print(df.dropna(axis=1))

# Only drop rows where all columns are NaN
print(df.dropna(how='all'))

# Drop rows that have fewer than 4 real values