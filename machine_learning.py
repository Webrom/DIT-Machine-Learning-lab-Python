import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Import Data
censusData = pd.read_csv("data/diabetic_data.csv", na_values=['?'])

# Select target label
targetLabels = censusData['readmitted']

# Take only numericals values
numeric_dfs = censusData.select_dtypes(include=['int64'])

# Drop id and patient id collumns
numeric_dfs = numeric_dfs.drop('encounter_id', axis=1)
numeric_dfs = numeric_dfs.drop('patient_nbr', axis=1)

# Take categories values
cat_dfs = censusData.select_dtypes(include=['object'])

# Drop the target values
cat_dfs = cat_dfs.drop('readmitted', axis=1)

# transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()

# convert to numeric encoding
vectorizer = DictVectorizer(sparse=False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))
