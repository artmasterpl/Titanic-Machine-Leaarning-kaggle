import pandas as pd
import utils
"""preprocessing can help with polynomials ( uses better graph curves)"""
from sklearn import linear_model, preprocessing

train = pd.read_csv("train.csv")
utils.clean_data(train)

"""graph """
target = train["Survived"].values
features_name = ["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]
features = train[features_name].values

""" classifiers """
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)
print(classifier_.score(features, target))

"""transform our linear into polynomial to get better result"""
poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(poly_features, target)
print(classifier_.score(poly_features, target))

