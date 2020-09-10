
# coding: utf-8

# # K-Means Clustering TASK 3

# # Importing the libraries

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# #  Importing the dataset

# In[15]:




dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, [1, 2]].values
Z =dataset.iloc[:, [3, 4]].values
print(X)
print("and")
print(Z)


# # Using the elbow method to find the optimal number of clusters ,  Spiral and petal

# In[16]:



from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Z)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# #  Training the K-Means model on the dataset for Spiral

# In[20]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# # Visualising the clusters Spiral

# In[19]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters ')
plt.xlabel('Spiral length)')
plt.ylabel('Spiral width')
plt.legend()
plt.show()


# # Training the K-Means model on the dataset for Petal

# In[21]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
z_kmeans = kmeans.fit_predict(Z)


# 
# # Visualising the clusters Petal

# In[22]:




plt.scatter(Z[z_kmeans == 0, 0], Z[z_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(Z[z_kmeans == 1, 0], Z[z_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(Z[z_kmeans == 2, 0], Z[z_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters ')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()
plt.show()

