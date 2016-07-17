
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix

# Method to create a classifier for decision tree
def classify(Features,Labels):
    classifier = tree.DecisionTreeClassifier('entropy')
    clf = classifier.fit(Features,Labels)
    return clf

f = open('Badges.txt')
i=0
badges,names,name_size,first_char,second_char = [],[],[],[],[]

for line in f:
    badge = line[0:2]
    badges.insert(i,badge)
    names.insert(i,line[2:len(line)-1])
    first_char.insert(i,line[2:3])
    name_size.insert(i,len(names[i]))
    second_char.insert(i,line[3:4])
    i=i+1
f.close()

j=0
training_badges,training_names,training_length,training_firstchar,training_secondchar=[],[],[],[],[]
test_badges,test_names,test_length,test_firstchar,test_secondchar=[],[],[],[],[]

while j<100:
    training_badges.insert(j,badges[j])
    training_names.insert(j,names[j])
    training_length.insert(j,name_size[j])
    training_firstchar.insert(j,first_char[j])
    training_secondchar.insert(j,second_char[j])
    j = j+1

while j<len(badges):
    test_badges.insert(j,badges[j])
    test_names.insert(j,names[j])
    test_length.insert(j,name_size[j])
    test_firstchar.insert(j,first_char[j])
    test_secondchar.insert(j,second_char[j])
    j = j+1

len(test_badges)

features_data = pd.DataFrame()
i=0

length_label,firstchar_label,secondchar_label=[],[],[]
vowels=('a','e','i','o','u')

while i<len(training_length):
    if (training_length[i]%2==0):
        length_label.insert(i,1)
    else:
        length_label.insert(i,0)
    if training_firstchar[i].lower().startswith(vowels):
        firstchar_label.insert(i,1)
    else:
        firstchar_label.insert(i,0)
    if training_secondchar[i].lower().startswith(vowels):
        secondchar_label.insert(i,1)
    else:
        secondchar_label.insert(i,0)
    i=i+1

features_data['Length Label']=length_label
features_data['FirstChar Label']=firstchar_label
features_data['SecondChar Label']=secondchar_label

clf=classify(features_data,training_badges)
clf

with open("badges.dot",'w') as f:
    f=tree.export_graphviz(clf,out_file=f)

test_len_label,test_fchar_label,test_schar_label=[],[],[]
i=0

while i<len(test_badges):
    if (test_length[i]%2==0):
        test_len_label.insert(i,1)
    else:
        test_len_label.insert(i,0)
    if test_firstchar[i].lower().startswith(vowels):
        test_fchar_label.insert(i,1)
    else:
        test_fchar_label.insert(i,0)
    if test_secondchar[i].lower().startswith(vowels):
        test_schar_label.insert(i,1)
    else:
        test_schar_label.insert(i,0)
    i=i+1

test_data = pd.DataFrame()
test_data['Length Label']=test_len_label
test_data['FirstChar Label']=test_fchar_label
test_data['SecondChar Label']=test_schar_label

test_results = []
test_results=clf.predict(test_data)
i,flag=0,0

while i<len(test_badges):
    if test_badges[i]!=test_results[i]:
        print "Not same for record number "+str(i)
        flag=flag+1
    i=i+1

if flag>0:
    print "Not predicted correctly"p
else:
    print "Predicted correctly"

cm=confusion_matrix(test_badges,test_results)

cm