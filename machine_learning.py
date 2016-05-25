import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Import Data
censusData = pd.read_csv("data/diabetic_data.csv", na_values=['?'])

# Drop id and patient id collumns
censusData = censusData.drop('encounter_id', axis=1)
censusData = censusData.drop('patient_nbr', axis=1)

# Draw plot
for col in censusData.columns:
    # if the column is continuous and that they have a cardinality higher than 10, histogram !
    if censusData[col].dtypes == 'int64' and censusData[col].value_counts().__len__() >= 10:
        tab = censusData[col].value_counts().sort_index()
        data = [
            go.Scatter(
                x=tab.keys(),
                y=tab.values
            )
        ]
        plot_url = py.plot(data, filename='data/html/' + col + ".html")
    # Else, it's just bar plot
    else:
        data = [
            go.Bar(
                x=censusData[col].value_counts().keys(),
                y=censusData[col].value_counts().values
            )
        ]
        plot_url = py.plot(data, filename='data/html/' + col + ".html")

# Select target label
targetLabels = censusData['readmitted']

# Take only numericals values
numeric_dfs = censusData.select_dtypes(include=['int64'])

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
