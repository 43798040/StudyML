from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data = load_iris()

X = data['data']
print(X)
print(X.shape)

Y = data['target']
print(Y)
print(Y.shape)

from sklearn.naive_bayes import GaussianNB
default = GaussianNB()
scores = cross_val_score(estimator=default, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())

from sklearn.neighbors import KNeighborsClassifier
default = KNeighborsClassifier()
scores = cross_val_score(estimator=default, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())
param = {'n_neighbors':range(2, 23, 4), 'algorithm':('auto', 'ball_tree', 'brute'),
         'weights':('uniform', 'distance'), 'metric':('euclidean', 'manhattan',
                                                      'chebyshev', 'hamming', 'jaccard', 'dice', 'russellrao')}

ModelEvaluation = GridSearchCV(default, param, scoring='accuracy', cv=5)
ModelEvaluation.fit(X, Y)
print(ModelEvaluation.best_params_)
print(ModelEvaluation.best_score_)


from sklearn.svm import SVC
default = SVC()
scores = cross_val_score(estimator=default, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())
param = {'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
         'degree':range(0, 10, 1)}

ModelEvaluation = GridSearchCV(default, param, scoring='accuracy', cv=5)
ModelEvaluation.fit(X, Y)
print(ModelEvaluation.best_params_)
print(ModelEvaluation.best_score_)

from sklearn.tree import DecisionTreeClassifier
default = DecisionTreeClassifier()
scores = cross_val_score(estimator=default, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())

from sklearn.linear_model import LogisticRegression
default = LogisticRegression()
scores = cross_val_score(estimator=default, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())