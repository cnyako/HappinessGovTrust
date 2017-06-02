import pandas as pd
import numpy as np
from matplotlib import pyplot
import sklearn.cluster as sk

dataset = pd.read_csv("happyGov.csv", sep=',', header=0)

#print(dataset)

x = dataset['Happiness']
y = dataset['GovTrust']

pyplot.scatter(x,y)
pyplot.xlabel("Happiness")
pyplot.ylabel("Trust in Government")

pyplot.show()

# Setup KMeans
X = dataset.as_matrix()

kmeans = sk.KMeans(n_clusters=4)

kmeans.fit(X)


cent = kmeans.cluster_centers_

labels = kmeans.labels_

colors = ["g.", "c.", "r.", "y."]

for i in range(len(X)):
	print("coordinate:",X[i], "label:", labels[i])
	pyplot.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

#pyplot.scatter(dataset['Happiness'], dataset['GovTrust'])
pyplot.scatter(cent[:,0], cent[:,1], marker = "x", s = 150, linewidths = 5, zorder = 10)
pyplot.title("Custers = 4")
pyplot.xlabel("Happiness")
pyplot.ylabel("Trust in Government")


pyplot.show()