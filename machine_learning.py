import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import tree
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Draw the confusion matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import math as math

# Import Data
censusData = pd.read_csv("data/diabetic_data.csv", na_values=['?', 'None'])

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

# Calcul of missing values
continous_dqr = numeric_dfs.describe().transpose()

categorical_dqr = cat_dfs.describe(include=['O']).transpose()

continous_dqr['Miss%'] = 0
categorical_dqr['Miss%'] = 0
categorical_dqr['Miss'] = 0
categorical_dqr['2nd Mode'] = 0
categorical_dqr['2nd Mode freq'] = 0

for col in censusData.columns:
    val = censusData[col].isnull().sum()
    count = censusData[col].value_counts(sort=1)

    if censusData[col].dtype != np.int64:
        categorical_dqr.ix[col, 'Miss%'] = (val / len(censusData[col])) * 100
        categorical_dqr.ix[col, 'Miss'] = val
        try:
            categorical_dqr.ix[col, '2nd Mode'] = count.index[1]
            categorical_dqr.ix[col, '2nd Mode freq'] = count.irow(1)
        except Exception:
            categorical_dqr.ix[col, '2nd Mode'] = 0
            categorical_dqr.ix[col, '2nd Mode freq'] = 0

    else:
        continous_dqr.ix[col, 'Miss%'] = (val / len(censusData[col])) * 100

        # Saving the DQRs

continous_dqr.to_csv("./data/DQR-ContinousFeatures.csv")
categorical_dqr.to_csv("./data/DQR-CategoricalFeatures.csv")

# Replace NaN values
for col in cat_dfs.columns:
    if categorical_dqr.loc[col, 'Miss%'] < 30:
        cat_dfs[col] = cat_dfs[col].fillna(categorical_dqr.loc[col, 'top'])
    elif 30 <= categorical_dqr.loc[col, 'Miss%'] < 60:
        firstValue = categorical_dqr.loc[col, 'top']
        secondValue = categorical_dqr.loc[col, '2nd Mode']
        threshold = categorical_dqr.loc[col, 'Miss'] * (categorical_dqr.loc[col, 'freq'] / (
        categorical_dqr.loc[col, 'freq'] + categorical_dqr.loc[col, '2nd Mode freq']))
        i = 0.0
        for row in range(len(cat_dfs[col])):
            if type(cat_dfs[col][row]) is float and math.isnan(cat_dfs[col][row]) and i < threshold:
                cat_dfs[col][row] = firstValue
                i += 1
            elif type(cat_dfs[col][row]) is float and math.isnan(cat_dfs[col][row]) and i >= threshold:
                cat_dfs[col][row] = secondValue
                i += 1
    elif categorical_dqr.loc[col, 'Miss%'] >= 60:
        cat_dfs = cat_dfs.drop(col, axis=1)

# To verify that we don't have missing value in categorical

categorical_dqr2 = cat_dfs.describe(include=['O']).transpose()

categorical_dqr2['Miss%'] = 0
categorical_dqr2['Miss'] = 0
categorical_dqr2['2nd Mode'] = 0
categorical_dqr2['2nd Mode freq'] = 0

for col in cat_dfs.columns:
    val = cat_dfs[col].isnull().sum()
    count = cat_dfs[col].value_counts(sort=1)

    categorical_dqr2.ix[col, 'Miss%'] = (val / len(cat_dfs[col])) * 100
    categorical_dqr2.ix[col, 'Miss'] = val
    try:
        categorical_dqr2.ix[col, '2nd Mode'] = count.index[1]
        categorical_dqr2.ix[col, '2nd Mode freq'] = count.irow(1)
    except Exception:
        categorical_dqr2.ix[col, '2nd Mode'] = 0
        categorical_dqr2.ix[col, '2nd Mode freq'] = 0


        # Saving the DQRs after remove (only to check)

categorical_dqr2.to_csv("./data/After_Remove_Missing_Values_DQR-CategoricalFeatures.csv")

# transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()

# convert to numeric encoding
vectorizer = DictVectorizer(sparse=False)
vec_cat_dfs = vectorizer.fit_transform(cat_dfs)

# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs))

######################Tree model using entropy criterion###############################

#define a decision tree model using entropy based information gain
decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')

#Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)

#fit the model using just the test set
decTreeModel.fit(instances_train, target_train)

#Use the model to make predictions for the test set queries
predictions = decTreeModel.predict(instances_test)


#Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
#plt.plot(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#--------------------------------------------
# Cross-validation to Compare to Models
#--------------------------------------------
print("------------------------")
print("Cross-validation Results")
print("------------------------")

#run a 5 fold cross validation on this model using the full census data
scores=cross_validation.cross_val_score(decTreeModel, instances_train, target_train, cv=5)
#the cross validaton function returns an accuracy score for each fold
print("Entropy based Model:")
print("Score by fold: " + str(scores))
#we can output the mean accuracy score and standard deviation as follows:
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")


"""
###########################Tree model using gini criterion#############################

# for a comparison we will do the same experiment using a decision tree that uses the Gini impurity metric
decTreeModel2 = tree.DecisionTreeClassifier(criterion='gini')

# Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels,
                                                                                               test_size=0.1,
                                                                                               random_state=0)

# fit the model using just the test set
decTreeModel2.fit(instances_train, target_train)

# Use the model to make predictions for the test set queries
predictions = decTreeModel2.predict(instances_test)

# Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

# Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
# plt.plot(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# --------------------------------------------
# Cross-validation to Compare to Models
# --------------------------------------------
print("------------------------")
print("Cross-validation Results")
print("------------------------")

# run a 5 fold cross validation on this model using the full census data
scores = cross_validation.cross_val_score(decTreeModel2, instances_train, target_train, cv=5)
print("Gini based Model:")
print("Score by fold: " + str(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))"""

#################Model forest random#############################
"""
#define a random forest model
rfc = RandomForestClassifier(n_estimators=100)

#Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)

#fit the model using just the test set
rfc.fit(instances_train, target_train)

#Use the model to make predictions for the test set queries
predictions = rfc.predict(instances_test)

#Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
#plt.plot(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#--------------------------------------------
# Cross-validation to Compare to Models
#--------------------------------------------
print("------------------------")
print("Cross-validation Results")
print("------------------------")

#run a 5 fold cross validation on this model using the full census data
scores=cross_validation.cross_val_score(rfc, instances_train, target_train, cv=5)
#the cross validaton function returns an accuracy score for each fold
print("Score by fold of the random forest: " + str(scores))
#we can output the mean accuracy score and standard deviation as follows:
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")
"""

#################Model nearest neighbour#############################

"""
#define a nearest neighbour model
nbrs = KNeighborsClassifier(n_neighbors=2, algorithm='auto')

#Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)

#fit the model using just the test set
nbrs.fit(instances_train, target_train)

#Use the model to make predictions for the test set queries
predictions = nbrs.predict(instances_test)

#Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
#plt.plot(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#--------------------------------------------
# Cross-validation to Compare to Models
#--------------------------------------------
print("------------------------")
print("Cross-validation Results")
print("------------------------")

#run a 5 fold cross validation on this model using the full census data
scores=cross_validation.cross_val_score(nbrs, instances_train, target_train, cv=5)
#the cross validaton function returns an accuracy score for each fold
print("Score by fold of the nearest neighbour: " + str(scores))
#we can output the mean accuracy score and standard deviation as follows:
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")

"""
