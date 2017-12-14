import pandas
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
data = pandas.read_csv("../data/pract_3.csv", header=None, sep=";", decimal=",")

print(data)

X = data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
print(X)
print(X.shape)

Y = data[[15]]
print(Y)
print(Y.shape)


# from sklearn.neighbors import KNeighborsClassifier
# default = KNeighborsClassifier()
# param = {'n_neighbors':range(2, 23, 4), 'algorithm':('auto', 'ball_tree', 'brute'),
#          'weights':('uniform', 'distance'), 'metric':('euclidean', 'manhattan',
#                                                       'chebyshev', 'hamming', 'jaccard', 'dice', 'russellrao')}
#
# ModelEvaluation = GridSearchCV(default, param, scoring='accuracy', cv=5)
# ModelEvaluation.fit(X, Y)
# print(ModelEvaluation.best_params_)
# print(ModelEvaluation.best_score_)

# {'algorithm': 'auto', 'metric': 'euclidean', 'n_neighbors': 10, 'weights': 'distance'}

from sklearn.neighbors import KNeighborsClassifier
default = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=10, weights='distance')
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()


# from sklearn.svm import SVC
# default = SVC()
# param = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
#          'degree':range(0, 10, 1)}
#
# ModelEvaluation = GridSearchCV(default, param, scoring='accuracy', cv=5)
# ModelEvaluation.fit(X, Y)
# print(ModelEvaluation.best_params_)
# print(ModelEvaluation.best_score_)

# {'degree': 0, 'kernel': 'rbf'}

from sklearn.svm import SVC
default = SVC(degree=0, kernel='rbf')
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()


# from sklearn.linear_model import LogisticRegression
# default = LogisticRegression()
# param = {'fit_intercept':(True, False), 'intercept_scaling':(True, False),
#          'class_weight':(None, 'balanced'), 'solver':('newton-cg', 'lbfgs', 'sag', 'saga'),
#          'multi_class':('ovr', 'multinomial')}
#
# ModelEvaluation = GridSearchCV(default, param, scoring='accuracy', cv=5)
# ModelEvaluation.fit(X, Y)
# print(ModelEvaluation.best_params_)
# print(ModelEvaluation.best_score_)

# {'class_weight': None, 'fit_intercept': True, 'intercept_scaling': True, 'multi_class': 'multinomial', 'solver': 'newton-cg'}

from sklearn.linear_model import LogisticRegression
default = LogisticRegression(class_weight=None, fit_intercept=True, intercept_scaling=True, multi_class='multinomial', solver='newton-cg')
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

# from sklearn.tree import DecisionTreeClassifier
# default = DecisionTreeClassifier()
# param = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'),
#          'max_depth':range(2, 10, 1), 'max_leaf_nodes':range(5, 50, 1)}
#
# ModelEvaluation = GridSearchCV(default, param, scoring='accuracy', cv=5)
# ModelEvaluation.fit(X, Y)
# print(ModelEvaluation.best_params_)
# print(ModelEvaluation.best_score_)

# {'criterion': 'entropy', 'max_depth': 8, 'max_leaf_nodes': 48, 'splitter': 'best'}

from sklearn.tree import DecisionTreeClassifier
default = DecisionTreeClassifier(criterion='entropy', max_depth=8, max_leaf_nodes=48, splitter='best')
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()


from sklearn.naive_bayes import GaussianNB
default = GaussianNB()
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()



knn = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=10, weights='distance')
tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, max_leaf_nodes=48, splitter='best')
log = LogisticRegression(class_weight=None, fit_intercept=True, intercept_scaling=True, multi_class='multinomial', solver='newton-cg')
from sklearn.ensemble import VotingClassifier
default = VotingClassifier(estimators=[('knn', knn), ('tree', tree), ('log', log)], voting='hard')
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

from sklearn.ensemble import VotingClassifier
default = VotingClassifier(estimators=[('knn', knn), ('tree', tree), ('log', log)], voting='soft')
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

from sklearn.ensemble import BaggingClassifier
default = BaggingClassifier(SVC(degree=0, kernel='rbf'))
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

from sklearn.ensemble import RandomForestClassifier
default = RandomForestClassifier()
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

from sklearn.ensemble import ExtraTreesClassifier
default = ExtraTreesClassifier()
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

from sklearn.ensemble import GradientBoostingClassifier
default = GradientBoostingClassifier()
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()

knn = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=10, weights='distance')
tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, max_leaf_nodes=48, splitter='best')
svc = SVC(degree=0, kernel='rbf')
log = LogisticRegression(class_weight=None, fit_intercept=True, intercept_scaling=True, multi_class='multinomial', solver='newton-cg')

# https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
from mlxtend.classifier import StackingClassifier
default = StackingClassifier([knn, tree, svc], meta_classifier=log, use_features_in_secondary=True)
accuracy = cross_val_score(estimator=default, X=X, y=Y, scoring='accuracy', cv=5)
print("accuracy:" + str(accuracy))
print("accuracy:" + str(accuracy.mean()))
print()
precision = cross_val_score(estimator=default, X=X, y=Y, scoring='precision_weighted', cv=5)
print("precision:" + str(precision))
print("precision:" + str(precision.mean()))
print()
recall = cross_val_score(estimator=default, X=X, y=Y, scoring='recall_weighted', cv=5)
print("recall:" + str(recall))
print("recall:" + str(recall.mean()))
print()
f1 = cross_val_score(estimator=default, X=X, y=Y, scoring='f1_weighted', cv=5)
print("f-score:" + str(f1))
print("f-score:" + str(f1.mean()))
print()