import pandas
import kfold_template

# from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("temperature_data.csv")

#Make categorical variables into dummies
dataset = pandas.get_dummies(dataset)

# print(dataset)

#shuffle dataset
dataset = dataset.sample(frac=1).reset_index()

# print(dataset)

#make a y variable, x variable

target = dataset["actual"].values
data = dataset.drop(["actual", "level_0"], axis = 1)

feature_list = data.columns
data = data.values

# print(feature_list)
# print(target)
# print(data)


# max_depth == 'how many times you chop'. The more you chop the more you reach a gini impurity of zero
# Rule of Thumb, start with a large max depth and go down. 

machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True)
# you estimate n number of times... which means n number of trees. You ask those trees to make predictions. 
# bootstrap is a technique of sampling from the sample itself, and with replacement. You draw, put it back, then draw again, n number of times. 
# if there is a majority of the trees making a prediction, we pick that as our answer. 

# kfold validation
return_values = kfold_template.run_kfold(machine, data, target, 4, True)

print(return_values)

machine = RandomForestClassifier(criterion="gini", max_depth=2, n_estimators=100, bootstrap = True)
## decrease max depth (10 -> 6). Run program for both, see which factors become less important. 

machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
# print(feature_importances_raw)
# # output is in the order of the columns listed
# print(feature_list)

feature_zip = zip(feature_list, feature_importances_raw)
feature_importances = [ (feature, round(importance, 4))	for feature, importance in feature_zip]
feature_importances = sorted(feature_importances, key = lambda x: x[1] )
# print(feature_importances)

[ print('{:14}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
#Output shows the importance of each variable
# 'Historical' has the highest importance value. Seems to be that Historical is a good predictor of Actual

# You want to have an accuracy score that is high and have the number of factors that are low. 









