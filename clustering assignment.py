# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 00:00:52 2020

@author: YASHESWI MISHRA
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 



crimedata=pd.read_csv("crime_data.csv")

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

crime_norm=norm_func(crimedata.iloc[:,1:])

##Find the optimal value of clusters(k)using scree plot(elbow method)####
k=list(range(2,16))
TWSS= []
for i in k:
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(crime_norm)
    WSS= []
    for j in range (i):
         WSS.append(sum(cdist(crime_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,crime_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)   

##Selected k=5 from scree plot ###
model= KMeans(n_clusters=5)    
model.fit(crime_norm)    
model.labels_    
cr=pd.Series(model.labels_)  
crimedata['cluster']=cr    
crimedata=crimedata.iloc[:,[5,0,1,2,3,4]]    
crimedata.iloc[:,[1,2,3,4,5]].groupby(crimedata.cluster).mean().plot(kind="bar")  
crimedata.cluster.value_counts()
##Inferences from the clusters:-
##The American states with a higher urban population in cluster 1 and 0 have highest assult rate and rape rate 
##On the contary, all the attibutes are lowest on the cluster 2 where the urban population is also the lowest
##Urban population is directly propotional to all the crime rates
## Higher the assault rate, higher will b the rape rate in these american states  
    