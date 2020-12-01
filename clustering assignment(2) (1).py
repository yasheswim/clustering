# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:31:47 2020

@author: YASHESWI MISHRA
"""

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
airplane=pd.read_excel("airlines.xlsx")

##Using k-means clustering method###plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
def norm_fun(i):
    x= (i-i.mean())/i.std()
    return (x)

##Remove unique id column
airplane.drop(["ID#"],inplace=True,axis=1)

airplane_norm=norm_fun(airplane)

##Model building###
k=list(range(2,20))
TWSS=[]
for i in k:
    WSS=[]
    kmeans1 = KMeans(n_clusters = i)
    kmeans1.fit(airplane_norm)
  # variable for storing within sum of squares for each cluster 
    for j in range (i):
      WSS.append(sum(cdist(airplane_norm.iloc[kmeans1.labels_==j,:],kmeans1.cluster_centers_[j].reshape(1,airplane_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'bx-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)     

###From the scree plot, we find the the optimal value of k is 10
model1=KMeans(n_clusters=8)  
model1.fit(airplane_norm)
model1.labels_
mf1=pd.Series(model1.labels_)
airplane["Clusters"]=mf1
airplane.iloc[:,[0]]
airplane=airplane.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
data_new=airplane.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].groupby(airplane.Clusters).mean()                    



####Hierarchal clustering method####
airplanedata=pd.read_excel("airlines.xlsx")
def norm_fun(i):
    x= (i-i.mean())/i.std()
    return (x)
airplanedata.drop(["ID#"],inplace=True,axis=1)

aeronorm=norm_fun(airplanedata)


aeronorm.describe()
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch                       

type(aeronorm)                       
newdata = linkage(aeronorm, method="ward",metric="euclidean")                       
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')                       
sch.dendrogram(
    newdata,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)                       
from sklearn.cluster import	AgglomerativeClustering 
h1_ward	=	AgglomerativeClustering(n_clusters=8,linkage='ward',affinity = "euclidean").fit(aeronorm) 

mydata=h1_ward.labels_ 
bc=pd.Series(mydata)                      
bc.value_counts()                       
airplanedata["Cluster1"]=bc  
airplane["Cluster_hier"]=bc                     
airplanedata=airplanedata.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]                       
airplanedata.groupby(airplanedata.Cluster1).mean()
airplane_new=pd.concat([bc,mf1],axis=1)##cluster numbers for the two methods of kmeans and agglomerative

##OBSERVATIONS FROM KMEANS CLUSTERS OF AIRLINES DATA####
##Balance--Number of miles eligible for award travel
##Qual_miles--Number of miles counted as qualifying for Topflight status
##cc1_miles--Number of miles earned with freq. flyer credit card in the past 12 months
##cc2_miles--Number of miles earned with Rewards credit card in the past 12 months:
##cc3_miles--Number of miles earned with Small Business credit card in the past 12 months:
##1 = under 5,000--
##2 = 5,000 - 10,000--
##3 = 10,001 - 25,000--
##4 = 25,001 - 50,000--
##5 = over 50,000--
##Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
##Bonus_trans--Number of non-flight bonus transactions in the past 12 months
##Flight_miles_--Number of flight miles in the past 12 months
##Flight_trans_12--Number of flight transactions in the past 12 months
##Days_since_enroll--Number of days since Enroll_date
##Award?--Dummy variable for Last_award (1=not null, 0=null)

##INFERENCES##
airplane.iloc[:,[11]].value_counts()
airplane.rename(columns={'Award?':'Award'},inplace=True)
pd.crosstab(airplane.cc1_miles,airplane.Clusters)
pd.crosstab(airplane.cc2_miles,airplane.Clusters)
pd.crosstab(airplane.cc3_miles,airplane.Clusters)

##0th cluster has one of the lowest award but top flier miles,this cluster
##is not doing frequent bookings on EastWest Airlines as it has very low transactions
##but its enrollment date is very high which means the customers in this cluster are 
##old but not opting for this airline so propositions must be made for this cluster of
##customers to make them opt for the service.

##1st cluster contains customers who are business-class fliers but have a considerably
##lower non-flight bonus transaction than other clusters.On the other hand,they have
##a decent record of miles earned on frequent flier credit card.So,these customers 
##can be termed as fliers who do not fly that frequently as enrollment time is also 
##4th lowest but when they do,they fly via business class by making use of frequent flier 
##credit card.

##2nd cluster clearly estimates that the customers are very non-frequent economic class
##fliers with the lowest balaance, quad_mile, less non flight transactions and least
##enrollment time.

##3rd cluster can be said to have vintage and old premium business class fliers with a high balance value 
##and a high quad_miles value. Their non-flight bonus transaction is also the highest
##and they have earned decent number of miles from frequent flier credit card.

##4th cluster customers can be termed as frequent economic class fliers having high 
##number of award earning miles who make frequent usage of flyer credit card and small
##business cards as they have one of the highest non flight transactions.

##5th cluster customers are also economic class fliers with lesser pooints making frequent
##use of reward credit card. They can be deemed as frequent travellers as they have high
##rate of miles travelled and transactions made. 

##6th cluster states travellers with high award earned miles as they have high amount of
##of miles earned on reward credit cards. Also, they make frequent non flight transactions
##and have miles earned on the same.They are the oldest customers according to the enrollment
##days

##7th cluster has the highest number of business class fliers with decent award points
## making frequent use of frequent flier credit card not making much use much use of 
##non flight transactions. So, one can target this cluster to encouage using more flight 
##reward cards to earn more points.















