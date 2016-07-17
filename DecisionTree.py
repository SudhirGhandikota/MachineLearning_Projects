
# coding: utf-8

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.externals.six import StringIO
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

accuracy,nodecount,samplecount,i=[],[],[],0
accuracy_val,j=[],0

def getClassifier(Features,Labels,sample_count):
    ##print 'Sample Count is '+str(sample_count)
    classifier = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=sample_count)
    clf = classifier.fit(Features,Labels)
    return clf

def getConfusionMatrix(clf,Features,Labels):
    predicted_results = clf.predict(Features)
    cm=confusion_matrix(Labels,predicted_results)
    stat=precision_recall_fscore_support(Labels,predicted_results)
    return cm,stat

def addtoplot(accu1,nd,sc,accu2):
    global accuracy
    global nodecount
    global samplecount
    global accuracy_val
    global nodecount_val
    global samplecount_val
    global i
    global j
    accuracy.insert(i,accu1) 
    nodecount.insert(i,nd)
    samplecount.insert(i,sc)
    i=i+1
    accuracy_val.insert(j,accu2) 
    j=j+1
    
def classify(Features,ClassLabels,ValidationFeatures,ValidationLabels,Sample_Count):
    clf = getClassifier(Features,ClassLabels,Sample_Count)
    cm,stat=getConfusionMatrix(clf,Features,ClassLabels)
    ##Precision and recall have to be extracted from the output array of train_test_split
    ## precision=stat[0][0] and rec=stat[1][0]
    ##train_test_split does not include accuracy so need to calculate it
    accu1 = float(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    print "CM of training data is"+str(cm)
    print 'Accuracy is '+str(accu1)+'precision is'+str(stat[0][0])+'recall is'+str(stat[1][0])
    cm,stat = getConfusionMatrix(clf,ValidationFeatures,ValidationLabels)
    accu2 = float(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    print "CM of validation/test data is"+str(cm)
    print 'Accuracy is '+str(accu2)+'precision is'+str(stat[0][0])+'recall is'+str(stat[1][0])
    addtoplot(accu1,clf.tree_.node_count,Sample_Count,accu2)
    ##print "added to plot list, node count is "+str(clf.tree_.node_count)+"Sample_Count is "+str(Sample_Count)

xlsfile = pd.ExcelFile('magic04.xlsx')
data_frame = xlsfile.parse('magic04.data')
len(data_frame)

d_training,d_testing=train_test_split(data_frame,test_size = 0.31545)
len(d_training)

d_validation,d_testing = train_test_split(d_testing,test_size=0.5)
len(d_validation)
len(d_testing)

TrainingFeatures = d_training[['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist']]
TrainingClassLabels = d_training['class']

TestFeatures = d_testing[['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist']]
TestLabels = d_testing['class']

ValidationFeatures = d_validation[['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist']]
ValidationLabels = d_validation['class']

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,1000)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,750)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,500)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,250)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,125)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,100)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,50)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,20)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,10)

classify(TrainingFeatures,TrainingClassLabels,ValidationFeatures,ValidationLabels,5)

plt.plot(samplecount,accuracy,label="Training")

plt.plot(samplecount,accuracy_val,label="Validation")
plt.xlabel('Min Leaf Nodes')
plt.ylabel('Accuracy of Training and validation')
plt.legend(bbox_to_anchor=(1.05, 1),borderaxespad=0.)
plt.show()

plt.plot(samplecount,nodecount)
plt.xlabel("Min Leav Nodes")
plt.ylabel('Number of Nodes')
plt.show()

classify(TrainingFeatures,TrainingClassLabels,TestFeatures,TestLabels,5)

classify(TrainingFeatures,TrainingClassLabels,TestFeatures,TestLabels,20)
nodecount