# Lab 2
from sklearn.datasets import load_iris

data = load_iris()

X = data['data']
print(X)
print(X.shape)

Y = data['target']
print(Y)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.tree import DecisionTreeClassifier
default_tree = DecisionTreeClassifier()
default_tree = default_tree.fit(X_train, Y_train)

Y_pred_train = default_tree.predict(X_train)
Y_pred_test = default_tree.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_train, Y_pred_train))
print(accuracy_score(Y_test, Y_pred_test))

feature_names = data['feature_names']
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=50)
classifier.fit(X, Y)
impotances = classifier.feature_importances_

from IPython.core.display import display
display(list(zip(feature_names, impotances)))
