#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using K-Means Clustering
# 
# In this project, we are given a dataframe with information on customer gender, age, annual income and their spending score. Out task is to cluster these customers into **meaningful clusters** so that we can increase their spending score by giving them appropritae offers. We will firs process the data, then do some data exploration, then we will determing **the optimal number of clusters and apply k-means clustering**. Finally, we will visualize these clusters and make sure that they make sense.

# In[56]:


# importing the dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[57]:


# loading the table
customer = pd.read_csv("Mall_Customers.csv")
customer.head()


# In[58]:


customer.shape


# In[59]:


customer.info()


# In[60]:


# checking for missing values
customer.isnull().sum()


# In[61]:


# Choosing annual income and spending score columns.
X = customer.iloc[:,[3,4]].values
X


# In[62]:


# calculating the within cluster sum of sqaures. The aim is to minimize this sum.
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[63]:


# Optimum number of clusters is 5.
# training the k means clustering model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

# return a label for each data point based on theri cluster.
y = kmeans.fit_predict(X)
y


# In[64]:


## visualizing all the clusters.
plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0], X[y==0,1], s= 50, c='green', label = 'Cluster1')
plt.scatter(X[y==1,0], X[y==1,1], s= 50, c='red', label = 'Cluster2')
plt.scatter(X[y==2,0], X[y==2,1], s= 50, c='blue', label = 'Cluster3')
plt.scatter(X[y==3,0], X[y==3,1], s= 50, c='pink', label = 'Cluster4')
plt.scatter(X[y==4,0], X[y==4,1], s= 50, c='brown', label = 'Cluster5')

# plotting the cluster centers.
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'cyan', label = 'centroids')

plt.title("Customer Groups")
plt.xlabel("Annual income")
plt.ylabel("Spending Score")


# We see a group with high annual income but very low spending score. If I am a business owner, I would definitely do promotions to attract these customers first.

# In[ ]:





# In[ ]:




