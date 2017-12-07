import pandas
data = pandas.read_csv("../data/data_banknote_authentication.txt", header=None, sep=",", decimal=".")
print(data)

X = data[[0, 1, 2, 3]]
print(X)
print(X.shape)

Y = data[[4]]
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



from sklearn.tree import DecisionTreeClassifier
two_tree = DecisionTreeClassifier(criterion='entropy', max_depth=50)
two_tree = two_tree.fit(X_train, Y_train)

Y_pred_train = two_tree.predict(X_train)
Y_pred_test = two_tree.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_train, Y_pred_train))
print(accuracy_score(Y_test, Y_pred_test))



from sklearn.tree import DecisionTreeClassifier
three_tree = DecisionTreeClassifier(criterion='entropy', max_depth=50, min_weight_fraction_leaf=0.4)
three_tree = three_tree.fit(X_train, Y_train)

Y_pred_train = three_tree.predict(X_train)
Y_pred_test = three_tree.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_train, Y_pred_train))
print(accuracy_score(Y_test, Y_pred_test))


from sklearn.tree import DecisionTreeClassifier
default_tree = DecisionTreeClassifier()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=default_tree, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())


from sklearn.tree import DecisionTreeClassifier
two_tree = DecisionTreeClassifier(criterion='entropy', max_depth=50)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=two_tree, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())


from sklearn.tree import DecisionTreeClassifier
three_tree = DecisionTreeClassifier(criterion='entropy', max_depth=50, min_weight_fraction_leaf=0.4)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=three_tree, X=X, y=Y, scoring="accuracy", cv=5)
print(scores)
print(scores.mean())

import graphviz
import sklearn.tree as tree
two_tree = DecisionTreeClassifier(criterion='entropy', max_depth=50)
two_tree.fit(X, Y)
dot_data = tree.export_graphviz(two_tree, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.02

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = data.iloc[:, pair]

    # Train
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=50).fit(X, Y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X.values[idx, 0], X.values[idx, 1], c=color,
                    cmap=plt.cm.RdBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.axis("tight")
plt.show()