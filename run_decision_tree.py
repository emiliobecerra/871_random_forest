import pandas
import kfold_template

from sklearn import tree

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
machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)


# kfold validation
return_values = kfold_template.run_kfold(machine, data, target, 4, True)

# print(return_values)

machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
# print(feature_importances_raw)
# # output is in the order of the columns listed
# print(feature_list)

feature_zip = zip(feature_list, feature_importances_raw)
feature_importances = [ (feature, round(importance, 4))	for feature, importance in feature_zip]
feature_importances = sorted(feature_importances, key = lambda x: x[1] )
print(feature_importances)

[ print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
#Output shows the importance of each variable
# 'Historical' has the highest importance value. Seems to be that Historical is a good predictor of Actual