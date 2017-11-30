# Практика №1 Feature Engineering + Clustering

import pandas
data = pandas.read_csv("../data/wholesale_customers_data.csv", header=0, sep=",", decimal=",")
print(data.columns)

# data = data.drop(['Channel', 'Region'], axis=1)
# print(data.columns)

print(data['Channel'].value_counts())

from IPython.display import display
display(data.head(5))

print(data['Channel'].value_counts(normalize=True))

print(data['Region'].value_counts())

data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].describe()

for feature in ['Channel', 'Region']:
    data[feature] = data[feature].astype('category')
data.describe(include=['category'])

# TODO only jupyter
data.describe(include='all').loc['count']

# TODO only jupyter
# import matplotlib.pyplot as pyplot
# import numpy
# ind = numpy.arange(data['Fresh'].count())
# pyplot.bar(ind, data['Fresh'])
# pyplot.show()
print(data[data['Fresh'] > 80000])
mean_for_fresh = data[(data['Fresh'] < 80000) & (data['Region'] == 3) & (data['Channel'] == 1)]['Fresh'].mean()
print(mean_for_fresh)
data.loc[data['Fresh'] > 80000, 'Fresh'] = mean_for_fresh
# TODO only jupyter
# pyplot.bar(ind, data['Fresh'])
# pyplot.show()

# TODO only jupyter
# ind = numpy.arange(data['Milk'].count())
# pyplot.bar(ind, data['Milk'])
# pyplot.show()
print(data[data['Milk'] > 30000])
mean_for_fresh = data[(data['Milk'] < 30000) & (data['Region'] == 3)]['Milk'].mean()
print(mean_for_fresh)
data.loc[data['Milk'] > 30000, 'Milk'] = mean_for_fresh
# TODO only jupyter
# pyplot.bar(ind, data['Milk'])
# pyplot.show()

# TODO only jupyter
# ind = numpy.arange(data['Grocery'].count())
# pyplot.bar(ind, data['Grocery'])
# pyplot.show()
print(data[data['Grocery'] > 40000])
mean_for = data[(data['Grocery'] < 40000) & (data['Channel'] == 2)]['Grocery'].mean()
print(mean_for)
data.loc[data['Grocery'] > 40000, 'Grocery'] = mean_for
# TODO only jupyter
# pyplot.bar(ind, data['Grocery'])
# pyplot.show()

# TODO only jupyter
# ind = numpy.arange(data['Frozen'].count())
# pyplot.bar(ind, data['Frozen'])
# pyplot.show()
print(data[data['Frozen'] > 20000])
mean_for = data[(data['Frozen'] < 20000) & (data['Channel'] == 1)]['Frozen'].mean()
print(mean_for)
data.loc[data['Frozen'] > 20000, 'Frozen'] = mean_for
# TODO only jupyter
# pyplot.bar(ind, data['Frozen'])
# pyplot.show()

# TODO only jupyter
# ind = numpy.arange(data['Detergents_Paper'].count())
# pyplot.bar(ind, data['Detergents_Paper'])
# pyplot.show()
print(data[data['Detergents_Paper'] > 20000])
mean_for = data[(data['Detergents_Paper'] < 20000) & (data['Channel'] == 2)]['Detergents_Paper'].mean()
print(mean_for)
data.loc[data['Detergents_Paper'] > 20000, 'Detergents_Paper'] = mean_for
# TODO only jupyter
# pyplot.bar(ind, data['Detergents_Paper'])
# pyplot.show()

# TODO only jupyter
# ind = numpy.arange(data['Delicassen'].count())
# pyplot.bar(ind, data['Delicassen'])
# pyplot.show()
print(data[data['Delicassen'] > 10000])
mean_for = data[(data['Delicassen'] < 10000) & (data['Region'] == 3)]['Delicassen'].mean()
print(mean_for)
data.loc[data['Delicassen'] > 10000, 'Delicassen'] = mean_for
# TODO only jupyter
# pyplot.bar(ind, data['Delicassen'])
# pyplot.show()

# TODO only jupyter
# pyplot.scatter(data['Fresh'], data['Milk'])
# pyplot.show()

# TODO only jupyter
# import seaborn
# seaborn.pairplot(data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']])
# pyplot.show()

quantitative = data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
display(quantitative.head(5))

X = quantitative.values
print(X)

from sklearn.preprocessing import StandardScaler
stand_scaler = StandardScaler().fit(X)
X_normalized = stand_scaler.transform(X)
print(X_normalized)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X_normalized)
X_reduced = pca.transform(X_normalized)
print(X_reduced)

# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1])
# pyplot.show()

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_reduced, method='ward', metric='euclidean')
# pyplot.figure()
# dn = dendrogram(Z)
# pyplot.show()

from sklearn.cluster import AgglomerativeClustering
lables = AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='complete').fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.metrics import silhouette_score
print(silhouette_score(X_reduced, lables, metric='manhattan'))

from sklearn.cluster import AgglomerativeClustering
lables = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.metrics import silhouette_score
print(silhouette_score(X_reduced, lables, metric='euclidean'))

from sklearn.cluster import AgglomerativeClustering
lables = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete').fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.metrics import silhouette_score
print(silhouette_score(X_reduced, lables, metric='euclidean'))

from sklearn.cluster import KMeans
lables = KMeans(n_clusters=4, max_iter=1000).fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.cluster import KMeans
lables = KMeans(n_clusters=6, max_iter=1000).fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.cluster import KMeans
lables = KMeans(n_clusters=7, max_iter=1000).fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.cluster import DBSCAN
lables = DBSCAN().fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.cluster import DBSCAN
lables = DBSCAN(eps=0.8).fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()

from sklearn.cluster import DBSCAN
lables = DBSCAN(metric='cosine').fit_predict(X_reduced)
# TODO only jupyter
# pyplot.scatter(X_reduced[:,0], X_reduced[:,1], c=lables)
# pyplot.show()