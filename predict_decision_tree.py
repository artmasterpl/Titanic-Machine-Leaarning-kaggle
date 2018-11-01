import pandas as pd
import utils
"""preprocessing can help with polynomials ( uses better graph curves)"""
from sklearn import tree, model_selection

train = pd.read_csv("train.csv")
utils.clean_data(train)

"""graph """
target = train["Survived"].values
features_name = ["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]
features = train[features_name].values

generalized_tree = tree.DecisionTreeClassifier(random_state=1)
generalized_tree_ = generalized_tree.fit(features, target)

print(generalized_tree_.score(features, target))
"""cv=50 how many time to do the operation"""
score = model_selection.cross_val_score(generalized_tree, features, target, scoring='accuracy', cv=50)

print(score)
print(score.mean())

generalized_tree = tree.DecisionTreeClassifier(
    random_state=1,
    max_depth=7,
    min_samples_split=2
    )
generalized_tree_ = generalized_tree.fit(features, target)

print(generalized_tree_.score(features, target))
score = model_selection.cross_val_score(generalized_tree, features, target, scoring='accuracy', cv=50)

print(score)
print(score.mean())
"""check the process of the decision tree and write into tree.dot file, see it in graphical representation using 
in linux dot -Tpng in windows use tree.gv extension and open in Graphviz program"""
tree.export_graphviz(generalized_tree_, feature_names=features_name, out_file="tree.gv")