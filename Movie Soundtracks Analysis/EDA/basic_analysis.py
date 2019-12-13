# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:27:21 2018

@author: youko
"""

##this script is to do exploratory analysis for cleaned_data.csv
import pandas as pd
import numpy as np
#from pd.Series import value_counts
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


##for ten or more attributes in this data, generate the mean, median, sd.
def basicstats(df,columns):
    '''this function is to generate the descriptive results
    '''
    meanlist=[]
    medianlist=[]
    stdlist=[]
    for column in columns:
        meanlist.append(df[column].mean())
        medianlist.append(df[column].median())
        stdlist.append(df[column].std())
    #combine the four descriptive lists
    description=pd.DataFrame(({'attributes':columns,'mean':meanlist,'median':
        medianlist,'standard_deviation':stdlist}))
    #print out the results to basic_stats.csv
    filename='basic_stats.csv'
    description.to_csv(filename, sep = ',',encoding='utf-8-sig', 
                       mode='w', index = False, float_format='%.8f')
    
      

##Part 2:  identify any attributes that may contain outliers 

# z-score to check for outliers
def ODzscore(df,scorecol,col):
    '''this function is to detect outliers with zscore
    '''
    from scipy.stats import zscore
    df[scorecol] = zscore(df[col])
    df["is_outlier"] = df[scorecol].apply(lambda x: x <= -5.5 or x >= 5.5)
    k=df[df["is_outlier"]]
    del df["is_outlier"]
    #return the outliers
    return k  

## DBSCAN part
def getdistplot(df,colname,figname,binsize):
    '''this function is to see the distribution of one specific column
    '''
    plt.figure(figsize=(8,6))
    sns.distplot(df[colname],bins=binsize,color='brown')
    plt.savefig(figname)

#check to see whether there are relationships 
    #between two columns
def seerelation(df,xcol,ycol,figname):
    '''this function is to see the effect xcol have on ycol 
    '''
    plt.figure(figsize=(12,6))
    sns.boxplot(x=xcol, y=ycol,data=df)
    plt.savefig(figname)


def getDistance(ODnorm,odcolumns,figname):
    '''this function is to get the best distance used for DBSCAN
    '''
    A = []
    B = []
    C = []
    # check for the best distance
    for i in np.linspace(0.1,5,20):
        db = DBSCAN(eps=i, min_samples=10).fit(ODnorm)

        cluster_number= len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    
        sum = 0
        for t in db.labels_:
            if t == -1: 
               sum = sum + 1
        C.append(sum)
    
        A.append(i)
        B.append(int(cluster_number))
        
    results = pd.DataFrame([A,B,C]).T
    results.columns = ['distance','Number of clusters','Number of outliers']
    results.plot(x='distance',y='Number of clusters',figsize=(10,6))
    plt.savefig(figname)
  
    
def ODdbscan(df,odcolumns,figname,e,m):
    ODdf=df[odcolumns]
    #normalize the data
    ODnorm = StandardScaler().fit_transform(ODdf)
    db = DBSCAN(eps=e, min_samples=m).fit(ODnorm)
    core = np.zeros_like(db.labels_, dtype=bool)
    core[db.core_sample_indices_] = True
    
    #get plot
    plt.figure(figsize=(16,12))
  

    unique_labels = set(db.labels_)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = [0, 0, 0, 1]
        
        outlier_member = (db.labels_ == -1)
        Outmember=np.where(outlier_member)[0]

        class_member = (db.labels_ == k)

        xy = ODnorm[class_member & core]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)

        xy = ODnorm[class_member & ~core]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)  
    
    plt.savefig(figname)
        
    df = df.join(pd.DataFrame(db.labels_))
    df = df.rename(columns={0:'Cluster'})
    clcount=df['Cluster'].value_counts()
    if clcount[-1]>0:
        return Outmember
    else:
        return 'No Outliers'

##bin the popularity of soundtracks and create a new column as to show its category
# we define a popularity level based on such criteria:
# popularity >= 65 : very popular; category number: 4
# 40<= popularity <65 : popular; category number: 3
# 15<= popularity <40 : common; category number: 2
# popularity < 15 : not popular; category number: 1

def poplevel (row):
    if row['popularity'] >=65 :
       return 4
    if row['popularity'] <65 and row['popularity']>=40 :
       return 3
    if row['popularity'] <40 and row['popularity']>=15 :
       return 2
    if row['popularity'] <15 :
       return 1
  
    return 'Other'

def main():
    myData=pd.read_csv('cleaned_data.csv',sep=',',encoding='utf-8-sig')
    #myData=myData[0:2000]
    ##get the basic descriptions for some attribtues
    columnnames=['acousticness','danceability',
 'duration_ms','energy','instrumentalness',
 'liveness','loudness','speechiness',
 'tempo','popularity','Movie_rate']
    basicstats(myData,columnnames)
    
    #check for outliers
    #first use zscore detection
    zscorecol=['acousticness','danceability','duration_ms','energy',
 'instrumentalness','key','liveness','loudness',
 'mode','speechiness','tempo','valence','popularity',
 'Movie_gross','Movie_runtime']
    #zscoreod to store the outliers detected
    zscoreod=[]
    for col in zscorecol:
        zscoreod.append(ODzscore(myData,'zs',col))
        del myData['zs']
    #print the zscore results
    with open ('zscoreOD.txt','w') as f:
        for i in range(0,len(zscorecol)):
        
            msg = "%s: %.f" % (zscorecol[i], len(zscoreod[i]))
            f.write(msg)
            f.write('\n')
    f.close()
    
    ######################################
    #check the outliers by DBSCAN
    #see the distribution of 'popularity'
    getdistplot(myData,'popularity','popularity_dist.png',4)
    
    #check to see whether there are relations between 'time_signature'
    #and 'popularity'
    seerelation(myData,'time_signature','popularity','time_pop.png')
    
    
    #first,use 'speechiness','tempo','loudness'
    #to detect outliers
    odcolumns=['tempo','loudness','speechiness']
    ODdf=myData[odcolumns]
    ODnorm=StandardScaler().fit_transform(ODdf)
    getDistance(ODnorm,odcolumns,'distance1.png')
    #after checking the distance, do DBSCAN
    dbres=ODdbscan(myData,odcolumns,'dbscan1.png',1,7)
    np.savetxt('dbscanOD.txt', dbres,fmt='%.f')
    ##bin the data
    myData['pop_level']  = myData.apply (lambda row: poplevel (row),axis=1)
    filename='binned_data.csv'
    myData.to_csv(filename, sep = ',',encoding='utf-8-sig', mode='w', index = False, float_format='%.8f')

if __name__ == "__main__":
	 main()
     
