import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Import Data
censusData = pd.read_csv("data/diabetic_data.csv", na_values=['?','None'])

# Drop id and patient id collumns
censusData = censusData.drop('encounter_id', axis=1)
censusData = censusData.drop('patient_nbr', axis=1)
# Select target label
targetLabels = censusData['readmitted']
# Drop the target values
censusData = censusData.drop('readmitted', axis=1)


# Draw plot, uncomment to use it
"""for col in censusData.columns:
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
        plot_url = py.plot(data, filename='data/html/' + col + ".html")"""



# Take only numericals values
numeric_dfs = censusData.select_dtypes(include=['int64'])

# Take categories values
cat_dfs = censusData.select_dtypes(include=['object'])

#Calcul of missing values
continous_dqr = numeric_dfs.describe().transpose()

categorical_dqr = cat_dfs.describe(include=['O']).transpose()

continous_dqr['Miss%'] = 0
categorical_dqr['2nd Mode'] = 0
categorical_dqr['2nd Mode freq'] = 0

for col in censusData.columns:
    val = censusData[col].isnull().sum()
    count = censusData[col].value_counts(sort=1)

    if censusData[col].dtype != np.int64:
        categorical_dqr.ix[col, 'Miss%'] = (val/len(censusData[col]))*100
        try:
            categorical_dqr.ix[col, '2nd Mode'] = count.index[1]
            categorical_dqr.ix[col, '2nd Mode freq'] = count.irow(1)
        except Exception:
            categorical_dqr.ix[col, '2nd Mode'] = 0
            categorical_dqr.ix[col, '2nd Mode freq'] = 0

    else:
        continous_dqr.ix[col, 'Miss%'] = (val/len(censusData[col]))*100

    # Saving the DQRs

continous_dqr.to_csv("./data/DQR-ContinousFeatures.csv")
categorical_dqr.to_csv("./data/DQR-CategoricalFeatures.csv")

#Replace NaN values
for col in cat_dfs.columns:
    if categorical_dqr.loc[col, 'Miss%'] < 30:
        cat_dfs[col] = cat_dfs[col].fillna(categorical_dqr.loc[col, 'top'])
    elif categorical_dqr.loc[col, 'Miss%'] > 60:
        cat_dfs = cat_dfs.drop(col, axis=1)




# transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()

# convert to numeric encoding
vectorizer = DictVectorizer(sparse=False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))
