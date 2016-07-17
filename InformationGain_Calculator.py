
# coding: utf-8
import pandas as pd
import numpy as np
import scipy.stats as sc
grades_sum
low_grades
med_grades
high_grades
students_sum

def entropy_calculator(label,label_list,grades_list):
    cal_entry=0
    global low_grades
    global med_grades
    global high_grades
    a_grade = freq_label_finder('A',label,label_list,grades_list)
    b_grade = freq_label_finder('B',label,label_list,grades_list)
    c_grade = freq_label_finder('C',label,label_list,grades_list)
    d_grade = freq_label_finder('D',label,label_list,grades_list)
    grades_sum = a_grade+b_grade+c_grade+d_grade
    if label=='Low':
        low_grades = grades_sum
        #print 'low grades '+str(low_grades)
    elif label=='Med':
        med_grades = grades_sum
        #print 'med grades '+str(med_grades)
    elif label=='High':
        high_grades = grades_sum
        #print 'high grades '+str(high_grades)
    #print 'Sum is '+str(sum_grade)
    prob_list = [float(a_grade)/float(grades_sum),float(b_grade)/float(grades_sum),float(c_grade)/float(grades_sum),float(d_grade)/float(grades_sum)]
    cal_entry = sc.entropy(prob_list,None,2)
    return cal_entry

def weighted_entropy_calculator(E_low,E_med,E_high):
    #print 'Student sum is '+str(students_sum)
    #print 'low grades are '+str(low_grades)
    E_w = ((float(low_grades)/float(students_sum))*(E_low)) + ((float(med_grades)/float(students_sum))*(E_med))+((float(high_grades)/float(students_sum))*(E_high))
    return E_w

def freq_finder(grade):
#used to get grade frequency of C++ grades to find dataset entropy
    freq_grade=0
    i=0
    
    while i<students_sum:
      if grades_c[i]==grade:
        freq_grade=freq_grade+1
      i = i+1
    return freq_grade
    
def freq_label_finder(grade,label,label_list,grades_list):
#used to find grade frequency related to a particular subject label
    freq_grade=0
    i=0
    
    while i<len(grades_list):
        #print 'in while grade. Grade is'+grade+' label is '+label
        if label_list[i]==label and grades_list[i]==grade:
            freq_grade=freq_grade+1
        i = i+1
    #print 'Freq is '+str(freq_grade)
    return freq_grade

xlsfile = pd.ExcelFile('DataHW1A.xlsx')
marks_frame = xlsfile.parse('Sheet1')
students_sum = len(marks_frame)

zscore_phy=sc.zscore(marks_frame['Phys'])
zscore_phy

phy_zlabel=[]
i=0

while i<len(zscore_phy):
    if zscore_phy[i]<-0.3:
        phy_zlabel.insert(i,'Low')
    if zscore_phy[i]>=-0.3 and zscore_phy[i]<=0.3:
        phy_zlabel.insert(i,'Med')
    if zscore_phy[i]>0.3:
        phy_zlabel.insert(i,'High')
    i = i+1

marks_frame['Zscore-phy']=zscore_phy
marks_frame['Phy-Label']=phy_zlabel
zscore_maths = sc.zscore(marks_frame['Maths'])
zscore_maths
maths_zlabel=[]

i=0

while i<len(zscore_maths):
    if zscore_maths[i]<-0.3:
        maths_zlabel.insert(i,'Low')
    if zscore_maths[i]>=-0.3 and zscore_maths[i]<=0.3:
        maths_zlabel.insert(i,'Med')
    if zscore_maths[i]>0.3:
        maths_zlabel.insert(i,'High')
    i = i+1

marks_frame['Zscore-Maths']=zscore_maths
marks_frame['Maths-label']=maths_zlabel

grades_c = marks_frame['C++ Grade']

##calculations related to finding entropy for physics subject

phy_Low_entropy = entropy_calculator('Low',phy_zlabel,grades_c)
phy_Med_entropy=  entropy_calculator('Med',phy_zlabel,grades_c)
phy_High_entropy= entropy_calculator('High',phy_zlabel,grades_c)
phy_wei_entropy = weighted_entropy_calculator(phy_Low_entropy,phy_Med_entropy,phy_High_entropy)
phy_wei_entropy

a_grade=freq_finder('A')
b_grade=freq_finder('B')
c_grade=freq_finder('C')
d_grade=freq_finder('D')
grades_prob = [float(a_grade)/40.0,float(b_grade)/40.0,float(c_grade)/40.0,float(d_grade)/40.0]
Entropy_dataset = sc.entropy(grades_prob,None,2)
Entropy_dataset
IGain_phy = Entropy_dataset - phy_wei_entropy
IGain_phy

##calculations related to finding entropy for math subject

math_Low_entropy = entropy_calculator('Low',maths_zlabel,grades_c)
math_Med_entropy = entropy_calculator('Med',maths_zlabel,grades_c)
math_High_entropy = entropy_calculator('High',maths_zlabel,grades_c)
math_wei_entropy = weighted_entropy_calculator(math_Low_entropy,math_Med_entropy,math_High_entropy)
math_wei_entropy
IGain_math = Entropy_dataset - math_wei_entropy
IGain_math

if IGain_math>IGain_phy:
    print 'Maths z-score labels are most suitable to predict C++ grades'
else:
    print 'Physics z-score labels are most suitable to predict C++ grades'

