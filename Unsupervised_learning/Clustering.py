import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns

df = pd.read_csv("Mall_Customers.csv")
x = df.iloc[:, [3, 4]].values

####### ----- K-means Clustering ------ #######

# find optimal number 0f clusters using elbow method
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('#Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init="k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

# visualization

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label = 'C1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label = 'C2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label = 'C3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='black', label = 'C4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label = 'C5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('K-Means clustering of Customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()


##### Hierarchial clustering: Agglomerative and Divisive

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method= 'ward'))

plt.title('Dendrogram')
plt.xlabel('#Customers')
plt.ylabel('Euclidean dist')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_hc = hc.fit_predict(x)
print(y_hc)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c='red', label = 'C1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c='blue', label = 'C2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c='green', label = 'C3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c='black', label = 'C4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, c='magenta', label = 'C5')
#plt.scatter(hc.cluster_centers_[:, 0], hc.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Hierachial clustering of Customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()


