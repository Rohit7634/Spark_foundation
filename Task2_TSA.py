#!/usr/bin/env python
# coding: utf-8

# # Task2

# In[15]:


#impoerting Libraries for this task
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


iris=pd.read_csv('iris.csv')
print("Data Imported Successfully")

iris.head(10)


# In[10]:


#finding the optimum number of cluster for k-means classification

x=iris.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans
wcs=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',
                 max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcs_iter=kmeans.inertia_
    wcs.append(wcs_iter)
    
#ploting
plt.plot(range(1,11),wcs)
plt.title('The elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()


# In[12]:


#applying kmeans dataset

kmeans=KMeans(n_clusters=3,init='k-means++',
             max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)


# In[13]:


#Visualizing the cluster data through scatter plot

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],
           s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-virginica')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],
           s=100,c='green',label='Centroids')

plt.legend()

