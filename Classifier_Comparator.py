
# coding: utf-8

from sklearn.svm import SVC
import csv
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

ID,Cthickness,Csize,Cshape,MAdhes,Ecsize,Bnuc,Chro,Nnuc,Mit,classlabel =[],[],[],[],[],[],[],[],[],[],[]
i=0

with open('CancerData.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        ID.insert(i,row[0])
        Cthickness.insert(i,row[1])
        Csize.insert(i,row[2])
        Cshape.insert(i,row[3])
        MAdhes.insert(i,row[4])
        Ecsize.insert(i,row[5])
        Bnuc.insert(i,row[6])
        Chro.insert(i,row[7])
        Nnuc.insert(i,row[8])
        Mit.insert(i,row[9])
        classlabel.insert(i,row[10])

data = pd.DataFrame()
data['ID']=ID
data['ClumpThickness']=Cthickness
data['Uniformity of Cell Size']=Csize
data['Uniformity of Cell Shape']=Cshape
data['Marginal Adhesion']=MAdhes
data['Epithelial Cell Size']=Ecsize
data['Bare Nucliei']=Bnuc
data['Bland Chromatin']=Chro
data['Normal Nuclei']=Nnuc
data['Mitoses']=Mit
data['Class Labels']=classlabel
data.groupby('Bare Nucliei').count()

# from sklearn.preprocessing import Imputer
# X=Imputer(missing_values=0,strategy='most_frequent')
# X.fit(test)
# converted_Bnuc=X.fit_transform(test) output is an array

i=0
while i<len(data):
    if data['Bare Nucliei'][i]=='?':
        data['Bare Nucliei'][i] = '1'
    i = i+1

data.groupby('Bare Nucliei').count()

d_training,d_testing=train_test_split(data,test_size = 0.2835)
len(d_training)

classifier = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=25)

data.head()

TrainingFeatures = d_training[['ClumpThickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
                           'Epithelial Cell Size','Bare Nucliei','Bland Chromatin','Normal Nuclei','Mitoses']]

TrainingLabels = d_training['Class Labels']

clf = classifier.fit(TrainingFeatures,TrainingLabels)

with open("DecisionTree.dot",'w') as f:
      f=tree.export_graphviz(clf,out_file=f)

TestFeatures = d_testing[['ClumpThickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
                           'Epithelial Cell Size','Bare Nucliei','Bland Chromatin','Normal Nuclei','Mitoses']]

TestLabels = d_testing['Class Labels']
predicted_features = clf.predict(TestFeatures)
cm=confusion_matrix(TestLabels,predicted_features)
cm

stat=precision_recall_fscore_support(TestLabels,predicted_features)
stat
precision = stat[0][0]
recall = stat[1][0]
accuracy = float(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
Fscore = stat[2][0]
print "Precision is "+str(precision)+" Recall is "+str(recall)+"Accuracy is "+str(accuracy)+" and Fscore is "+str(Fscore)

classifier2 = SVC()
classifier2.fit(TrainingFeatures,TrainingLabels)
svm_predictions = classifier2.predict(TestFeatures)
cm_svm=confusion_matrix(TestLabels,svm_predictions)
cm_svm

stat_svm=precision_recall_fscore_support(TestLabels,svm_predictions)
stat_svm
precision_svm = stat_svm[0][0]
recall_svm = stat_svm[1][0]
Fscore_svm = stat_svm[2][0]
accuracy_svm = float(cm_svm[0][0]+cm_svm[1][1])/(cm_svm[0][0]+cm_svm[0][1]+cm_svm[1][0]+cm_svm[1][1])
print "Precision is "+str(precision_svm)+" Recall is "+str(recall_svm)+"Accuracy is "+str(accuracy_svm)+" and Fscore is "+str(Fscore_svm)

i,j,k=0,0,0
Csize_4,Bnuc_4,Csize_2,Bnuc_2=[],[],[],[]

while i<len(data):
    if data['Class Labels'][i]=='4':
        Csize_4.insert(j,data['Uniformity of Cell Size'][i])
        Bnuc_4.insert(j,data['Bare Nucliei'][i])
        j=j+1
    if data['Class Labels'][i]=='2':
        Csize_2.insert(k,data['Uniformity of Cell Size'][i])
        Bnuc_2.insert(k,data['Bare Nucliei'][i])
        k=k+1
    i = i+1

plt.plot(Csize_4,Bnuc_4,'bs',Csize_2,Bnuc_2,'g^')
plt.xlabel("Uniform Cell Size")
plt.ylabel("Bare Nucliei")
plt.show()

predicted_features
TestLabels
DataFeatures = data[['ClumpThickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
                           'Epithelial Cell Size','Bare Nucliei','Bland Chromatin','Normal Nuclei','Mitoses']]

def getNearestLabels(k,index):
    i=0
    neighbors = []
    nbrs = NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(DataFeatures)
    distances,indices = nbrs.kneighbors(DataFeatures)
    u=data[index:index+1].values
    print "Actual label is "+u[0][10]
    while i < k-1:
        neighbor = indices[index][i+1]
        l = data[neighbor:neighbor+1].values;
        print "ID of neighbor number "+l[0][0]+" is "+str(neighbor)+" Label is "+l[0][10]
        i = i+1

getNearestLabels(4,87)

getNearestLabels(2,87)

getNearestLabels(6,87)

getNearestLabels(8,87)