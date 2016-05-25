import pandas as pd

#Import Data
censusData = pd.read_csv("data/diabetic_data.csv", na_values=['?'])

#Select target label
targetLabels = censusData['readmitted']

#Take only numericals values
numeric_dfs = censusData.select_dtypes(include=['int64'])

#Take categories values
cat_dfs = censusData.select_dtypes(include=['object'])

#Drop the target values
cat_dfs = cat_dfs.drop('readmitted',axis=1)