
# coding: utf-8

import pandas as pd
import numpy as np
import sys

##using pandas module to read the excel file

xlsfile = pd.ExcelFile('DataHW1A.xlsx')

##parsing the sheet that we want to extract data from

marks_frame = xlsfile.parse('Sheet1')

##using assign method to create new column from existing columns

students_marks_width= marks_frame.assign(Total = marks_frame['Phys']+marks_frame['Maths'])
students_marks_width.sort(['Total'],ascending=[0])
copy_student_marks = students_marks_width.copy()
width_bins = pd.cut(students_marks_width['Total'],5)
pd.value_counts(width_bins)

width_grades_list=pd.cut(students_marks_width['Total'],5,labels = ['F','D','C','B','A'])
pd.value_counts(width_grades_list)
students_marks_width['Grade']=width_grades_list
students_marks_width
total_list = copy_student_marks['Total']
freq_grades_list = []

i = 0
while i < len(total_list):
 if i<8:
    freq_grades_list.insert(i,'A')
 elif i<16:
    freq_grades_list.insert(i,'B')
 elif i<24:
    freq_grades_list.insert(i,'C')
 elif i<32:
    freq_grades_list.insert(i,'D')
 elif i<40:
    freq_grades_list.insert(i,'F')
 i = i+1

Marks_Frame = copy_student_marks.sort(['Total'],ascending=[0])
Marks_Frame['Grade'] = freq_grades_list
students_marks_freq = Marks_Frame.sort('Student id')
students_marks_freq

final_freq_grades = students_marks_freq['Grade']
final_width_grades = students_marks_width['Grade']

if(len(final_freq_grades)!=len(final_width_grades)):
    print "Not Same"
    Sys.exit(1)
    
i=0

while (i < len(final_freq_grades)):
    if (final_freq_grades[i]>final_width_grades[i]):
        print 'Student '+str(i+1)+' got '+'less grade '+final_freq_grades[i]+' in freq binning when compared to '+final_width_grades[i]+' in width binning'
    if (final_freq_grades[i]<final_width_grades[i]):
        print 'Student '+str(i+1)+' got '+'better grade '+final_freq_grades[i]+' in freq binning when compared to '+final_width_grades[i]+' in width binning'
    i=i+1



