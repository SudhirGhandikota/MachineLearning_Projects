
# coding: utf-8

import numpy as py
import pandas as pd
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches

points=py.array([[6,12],[19,7],[15,4],[11,0],[18,12],[9,20],[19,22],[18,17],[5,11],[4,18],[7,15],[21,18],[1,19],[1,4],[0,9],[5,11]])

x_dbscan=[1,3,5,6,8,11,12,13,14,15,16,22,28,32,33,34,35,36,37,42,58]
dbscan_points=py.array(zip(x_dbscan,py.zeros(len(x_dbscan))),dtype=py.int)

points_rev = points[::-1]

cluster1 = py.array([[0,0]])
cluster2 = py.array([[0,0]])
cluster3 = py.array([[0,0]])
cluster4 = py.array([[0,0]])
clusters=[]

def create_dendrogram(dist):
    global points
    distances=linkage(points,dist)
    c,coph_dists=cophenet(distances,pdist(points))
    plt.figure(figsize=(25,10))
    plt.title('Dendogram')
    plt.xlabel('Points')
    plt.ylabel('Distance')
    dend=dendrogram(distances,show_contracted=True)
    plt.show()
    dend2=dendrogram(distances,show_contracted=True,truncate_mode='lastp',p=3)
    plt.show()
    clusters=fcluster(distances,3,criterion='maxclust')
    return c,clusters

def get_SSE(cluster,cent):
    sse=0
    i=0
    while i<len(cluster):
        dist=sc.euclidean(cluster[i],cent)
        sse=sse+dist
        print 'sse is'+str(sse)
        i=i+1

def get_correlation(cluster):
    global points
    proximity_matrix=pairwise_distances(points,metric='euclidean')
    incidence_matrix = py.ones((points.size/2,points.size/2))
    i,j=0,0
    while i<points.size/2:
        j=0
        while j<points.size/2:
            if cluster[i]!=cluster[j]:
                incidence_matrix[i][j]=0
            j=j+1
        i=i+1
    corr=py.corrcoef(proximity_matrix,incidence_matrix)
    return corr,proximity_matrix,incidence_matrix

def get_dend_xy(clusters):
    global points
    i=0
    cluster1 = py.array([[0,0]])
    cluster1 = py.delete(cluster1,0,0)
    cluster2 = py.array([[0,0]])
    cluster2 = py.delete(cluster2,0,0)
    cluster3 = py.array([[0,0]])
    cluster3 = py.delete(cluster3,0,0)
    c1_x,c1_y,c2_x,c2_y,c3_x,c3_y=[],[],[],[],[],[]
    while i<len(clusters):
        if clusters[i]==1:
            c1_x.insert(i,points[i][0])
            c1_y.insert(i,points[i][1])
            cluster1=py.insert(cluster1,(cluster1.size)/2,points[i],axis=0)
        if clusters[i]==2:
            c2_x.insert(i,points[i][0])
            c2_y.insert(i,points[i][1])
            cluster2=py.insert(cluster2,(cluster2.size)/2,points[i],axis=0)
        if clusters[i]==3:
            c3_x.insert(i,points[i][0])
            c3_y.insert(i,points[i][1])
            cluster3=py.insert(cluster3,(cluster3.size)/2,points[i],axis=0)
        i=i+1
    return cluster1,cluster2,cluster3,c1_x,c1_y,c2_x,c2_y,c3_x,c3_y

def initialize_clusters():
    global cluster1,cluster2,cluster3,cluster4
    cluster1 = py.array([[0,0]])
    cluster1 = py.delete(cluster1,0,0)
    cluster2 = py.array([[0,0]])
    cluster2 = py.delete(cluster2,0,0)
    cluster3 = py.array([[0,0]])
    cluster3 = py.delete(cluster3,0,0)
    cluster4 = py.array([[0,0]])
    cluster4 = py.delete(cluster4,0,0)

def insert_cluster(i,cname,points):
    global cluster1,cluster2,cluster3,cluster4,clusters
    clusters.insert(i,cname)
    if cname=='C1':
        cluster1=py.insert(cluster1,(cluster1.size)/2,points[i],axis=0)
    if cname=='C2':
        cluster2=py.insert(cluster2,(cluster2.size)/2,points[i],axis=0)
    if cname=='C3':
        cluster3=py.insert(cluster3,(cluster3.size)/2,points[i],axis=0)
    if cname=='C4':
        cluster4=py.insert(cluster4,(cluster4.size)/2,points[i],axis=0)

def get_cluster_xy(cluster):
    i=0
    cluster_x,cluster_y=[],[]
    while i<len(cluster):
        cluster_x.insert(i,cluster[i][0])
        cluster_y.insert(i,cluster[i][1])
        i=i+1
    return cluster_x,cluster_y

def bsas(points):
    global cluster1,cluster2,cluster3,cluster4,clusters,numclusters
    initialize_clusters()
    clusters=[]
    theta = 12
    cent1=0
    clusters.insert(0,'C1')
    numclusters=1
    i=1
    cent1 = points[0]
    cluster1 = py.insert(cluster1,(cluster1.size)/2,points[0],axis=0)
    while i<16:
        #let initial centre be the first point
        if numclusters==4:
            dist1 = abs(sc.euclidean(points[i],cent1))
            dist2 = abs(sc.euclidean(points[i],cent2))
            dist3 = abs(sc.euclidean(points[i],cent3))
            dist4 = abs(sc.euclidean(points[i],cent4))
            print 'dist 1 between '+str(points[i])+'and centre 1 '+str(cent1)+'is'+str(dist1)
            print 'dist 2 between'+str(points[i])+'and centre 2 '+str(cent2)+'is'+str(dist2)
            print 'dist 3 between'+str(points[i])+'and centre 3 '+str(cent3)+'is'+str(dist3)
            print 'dist 4 between'+str(points[i])+'and centre 4 '+str(cent4)+'is'+str(dist4)
            if dist1<=theta and dist1<dist2 and dist1<=dist3 and dist1<=dist4:
                insert_cluster(i,'C1',points)
                cent1 = cluster1.mean(axis=0)
                i=i+1
                continue
            if dist2<=theta and dist2<=dist3 and dist2<=dist4:
                insert_cluster(i,'C2',points)
                cent2 = cluster2.mean(axis=0)
                i=i+1
                continue
            if dist3<=theta and dist3<=dist4:
                insert_cluster(i,'C3',points)
                cent3 = cluster3.mean(axis=0)
                i=i+1
                continue
            if dist4<=theta:
                insert_cluster(i,'C4',points)
                cent4 = cluster4.mean(axis=0)
                i=i+1
                continue
            if dist1 and dist2 and dist3 and dist4>theta:
                print 'outlier found '+str(i)
                cent4=points[i]
                insert_cluster(i,'C4',points)
                numclusters = numclusters+1
                i=i+1
                continue
            
        if numclusters==3:
            dist1 = abs(sc.euclidean(points[i],cent1))
            dist2 = abs(sc.euclidean(points[i],cent2))
            dist3 = abs(sc.euclidean(points[i],cent3))
            print 'dist 1 between '+str(points[i])+'and centre 1 '+str(cent1)+'is'+str(dist1)
            print 'dist 2 between'+str(points[i])+'and centre 2 '+str(cent2)+'is'+str(dist2)
            print 'dist 3 between'+str(points[i])+'and centre 3 '+str(cent3)+'is'+str(dist3)
            if dist1<=theta and dist1<=dist2 and dist1<=dist3:
                insert_cluster(i,'C1',points)
                cent1 = cluster1.mean(axis=0)
                i=i+1
                continue
            if dist2<=theta and dist2<=dist3:
                insert_cluster(i,'C2',points)
                cent2 = cluster2.mean(axis=0)
                i=i+1
                continue
            if dist3<=theta:
                insert_cluster(i,'C3',points)
                cent3 = cluster3.mean(axis=0)
                i=i+1
                continue
            if dist1 and dist2 and dist3>theta:
                print 'cluster 4 centre is '+str(i)
                cent4=points[i]
                insert_cluster(i,'C4',points)
                numclusters = numclusters+1
                i=i+1
                continue
            
        if numclusters==2:
            dist1 = abs(sc.euclidean(points[i],cent1))
            dist2 = abs(sc.euclidean(points[i],cent2))
            print 'dist 1 between '+str(points[i])+'and centre 1 '+str(cent1)+'is'+str(dist1)
            print 'dist 2 between'+str(points[i])+'and centre 2 '+str(cent2)+'is'+str(dist2)
            if dist1<=theta and dist1<=dist2:
                insert_cluster(i,'C1',points)
                cent1 = cluster1.mean(axis=0)
                i=i+1
                continue
            if dist2<=theta:
                insert_cluster(i,'C2',points)
                cent2 = cluster2.mean(axis=0)
                i=i+1
                continue
            if dist1 and dist2>theta:
                print 'cluster 3 centre is '+str(i)
                cent3=points[i]
                insert_cluster(i,'C3',points)
                numclusters = numclusters+1
                i=i+1
                continue
            
        if numclusters==1:
            dist1 = abs(sc.euclidean(points[i],cent1))
            print 'dist 1 between'+str(points[i])+'and centre 1 '+str(cent1)+'is'+str(dist1)
            if dist1<theta:
                insert_cluster(i,'C1',points)
                cent1 = cluster1.mean(axis=0)
                i=i+1
                continue
            if dist1>theta:
                print 'cluster 2 centre is '+str(i)
                cent2=points[i]
                insert_cluster(i,'C2',points)
                numclusters = numclusters+1
                i=i+1
                continue
    
def plot_clusters(x1,y1,x2,y2,x3,y3,x4,y4):
    global cluster1,cluster2,cluster3
    plt.plot(x1,y1,'bs',x2,y2,'g^',x3,y3,'ro',x4,y4,'ys',lw=2.)
    fig=plt.gcf()
    circle1 = plt.Circle(cluster1.mean(axis=0),6.5,fill=False)
    circle2 = plt.Circle(cluster2.mean(axis=0),7.5,fill=False)
    circle3 = plt.Circle(cluster3.mean(axis=0),4,fill=False)
    circle4 = plt.Circle(cluster4.mean(axis=0),3,fill=False)
    
    match1=mpatches.Patch(color='blue',label='Cluster 1')
    match2=mpatches.Patch(color='green',label='Cluster 2')
    match3=mpatches.Patch(color='red',label='Cluster 3')
    match4=mpatches.Patch(color='yellow',label='Cluster 4')
    plt.legend(handles=[match1,match2,match3,match4])
    
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    fig.gca().add_artist(circle3)
    fig.gca().add_artist(circle4)
    plt.show()

def plot_clusters_rev(x1,y1,x2,y2,x3,y3,x4,y4):
    global cluster1,cluster2,cluster3
    plt.plot(x1,y1,'bs',x2,y2,'g^',x3,y3,'ro',x4,y4,'ys',lw=2.)
    fig=plt.gcf()
    circle1 = plt.Circle(cluster1.mean(axis=0),10,fill=False)
    circle2 = plt.Circle(cluster2.mean(axis=0),7,fill=False)
    circle3 = plt.Circle(cluster3.mean(axis=0),8,fill=False)
    circle4 = plt.Circle(cluster4.mean(axis=0),4,fill=False)
    
    match1=mpatches.Patch(color='blue',label='Cluster 1')
    match2=mpatches.Patch(color='green',label='Cluster 2')
    match3=mpatches.Patch(color='red',label='Cluster 3')
    match4=mpatches.Patch(color='yellow',label='Cluster 4')
    plt.legend(handles=[match1,match2,match3,match4])
    
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    fig.gca().add_artist(circle3)
    fig.gca().add_artist(circle4)
    plt.show()

bsas(points)
clustering_1=clusters
c1_x,c1_y = get_cluster_xy(cluster1)
c2_x,c2_y = get_cluster_xy(cluster2)
c3_x,c3_y = get_cluster_xy(cluster3)
c4_x,c4_y = get_cluster_xy(cluster4)
plot_clusters(c1_x,c1_y,c2_x,c2_y,c3_x,c3_y,c4_x,c4_y)

bsas(points_rev)
clustering_2=clusters
c1_x,c1_y = get_cluster_xy(cluster1)
c2_x,c2_y = get_cluster_xy(cluster2)
c3_x,c3_y = get_cluster_xy(cluster3)
c4_x,c4_y = get_cluster_xy(cluster4)

plot_clusters_rev(c1_x,c1_y,c2_x,c2_y,c3_x,c3_y,c4_x,c4_y)
adjusted_rand_score(clustering_2,clustering_1)

c,clusters_single=create_dendrogram('single')
c
clusters_single

cluster1,cluster2,cluster3,c1_x,c1_y,c2_x,c2_y,c3_x,c3_y=get_dend_xy(clusters_single)
cent1=cluster1.mean(axis=0)
cent2=cluster2.mean(axis=0)
cent3=cluster3.mean(axis=0)

plt.plot(c1_x,c1_y,'bs',c2_x,c2_y,'g^',c3_x,c3_y,'ro',lw=2.)
fig=plt.gcf()
circle1 = plt.Circle(cent1,10,fill=False)
circle2 = plt.Circle(cent2,10.5,fill=False)
circle3 = plt.Circle(cent3,4,fill=False)

match1=mpatches.Patch(color='blue',label='Cluster 1')
match2=mpatches.Patch(color='green',label='Cluster 2')
match3=mpatches.Patch(color='red',label='Cluster 3')
plt.legend(handles=[match1,match2,match3])
    
fig.gca().add_artist(circle1)
fig.gca().add_artist(circle2)
fig.gca().add_artist(circle3)
plt.show()

sse1=get_SSE(cluster1,cent1)
sse2=get_SSE(cluster2,cent2)
sse3=get_SSE(cluster3,cent3)

c,clusters_complete=create_dendrogram('complete')
cluster1,cluster2,cluster3,c1_x,c1_y,c2_x,c2_y,c3_x,c3_y=get_dend_xy(clusters_complete)

cent1=cluster1.mean(axis=0)
cent2=cluster2.mean(axis=0)
cent3=cluster3.mean(axis=0)
cent3

plt.plot(c1_x,c1_y,'bs',c2_x,c2_y,'g^',c3_x,c3_y,'ro',lw=2.)
fig=plt.gcf()
circle1 = plt.Circle(cent1,4,fill=False)
circle2 = plt.Circle(cent2,8,fill=False)
circle3 = plt.Circle(cent3,10,fill=False)

match1=mpatches.Patch(color='blue',label='Cluster 1')
match2=mpatches.Patch(color='green',label='Cluster 2')
match3=mpatches.Patch(color='red',label='Cluster 3')
plt.legend(handles=[match1,match2,match3])
 
fig.gca().add_artist(circle1)
fig.gca().add_artist(circle2)
fig.gca().add_artist(circle3)
plt.show()

sse1=get_SSE(cluster1,cent1)
sse2=get_SSE(cluster2,cent2)
sse3=get_SSE(cluster3,cent3)

corr,proximity_matrix,incidence_matrix=get_correlation(clusters_single)
corr_matrix,proximity_matrix,incidence_matrix=get_correlation(clusters_complete)



dbscan_4=DBSCAN(eps=4,min_samples=3).fit(dbscan_points)
dbscan_4
labels_4=dbscan_4.labels_
core_points_4=dbscan_4.core_sample_indices_
core_points_4

dbscan_6=DBSCAN(eps=6,min_samples=3).fit(dbscan_points)
core_points_6=dbscan_6.core_sample_indices_
core_points_6

labels_4
core_points_4

labels_6=dbscan_6.labels_
labels_6

adjusted_rand_score(labels_4,labels_6)
dbscan_4.fit_predict(dbscan_points)

dbscan_points[:,0]
C0,C1,noise=[],[],[]

i=0
while i<len(labels_4):
    if labels_4[i]==1:
        C1.insert(i,dbscan_points[i][0])
    if labels_4[i]==0:
        C0.insert(i,dbscan_points[i][0])
    if labels_4[i]==-1:
        noise.insert(i,dbscan_points[i][0])
    i=i+1
    
C1
C0
noise

plt.plot(C0,py.zeros(len(C0)),'bs',C1,py.zeros(len(C1)),'g^',noise,py.zeros(len(noise)),'rs')
match1=mpatches.Patch(color='blue',label='Cluster 0')
match2=mpatches.Patch(color='green',label='Cluster 1')
match3=mpatches.Patch(color='red',label='Noise')
plt.legend(handles=[match1,match2,match3])
plt.show()

C0,noise=[],[]
i=0
while i<len(labels_6):
    if labels_4[i]==0:
        C0.insert(i,dbscan_points[i][0])
    if labels_4[i]==-1:
        noise.insert(i,dbscan_points[i][0])
    i=i+1

plt.plot(C0,py.zeros(len(C0)),'g^',noise,py.zeros(len(noise)),'rs')
match1=mpatches.Patch(color='blue',label='Cluster 0')
match2=mpatches.Patch(color='red',label='Noise')
plt.legend(handles=[match1,match2])
plt.show()