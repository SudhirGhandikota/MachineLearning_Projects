import pandas as pd
import numpy as np
import csv
import urllib2
from bs4 import BeautifulSoup
from lxml import etree
import openpyxl
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import scipy.stats as sc

raw_data = pd.DataFrame()
raw_data_2012 = pd.DataFrame()
classifiers1 = BaggingClassifier(DecisionTreeClassifier(),n_estimators=10,max_features=1.0,max_samples=1.0,bootstrap=True,oob_score=True)
classifiers2 = BaggingClassifier(KNeighborsClassifier(n_neighbors=7),n_estimators=10,max_features=1.0,max_samples=1.0,bootstrap=True,oob_score=True)
classifiers3 = BaggingClassifier(SVC(kernel='rbf',probability=True),n_estimators=10,max_features=1.0,max_samples=1.0,bootstrap=True,oob_score=True)

def get_race_data(places_data,state):
    i=0
    race = []
    global raw_data
    while i<len(places_data):
        maximum = max(places_data['%white'][i],places_data['%Black'][i],places_data['%ASIAN'][i],places_data['%HISP'][i])
        if maximum==places_data['%white'][i]:
            race.insert(i,'White')
        elif maximum==places_data['%Black'][i]:
            race.insert(i,'Black')
        elif maximum==places_data['%ASIAN'][i]:
            race.insert(i,'Asian')
        elif maximum==places_data['%HISP'][i]:
            race.insert(i,'Hisp')
        i=i+1
    
    places_data['Dominant Race']=race
    places_check = places_data[places_data.State==state]
    places_check = places_check.reset_index()
    i,j=0,0
    contbr_places=[]
    
    while i<len(raw_data):
        j=0
        while j<len(places_check):
            if raw_data['contbr_city'][i].lower()==places_check['Place'][j].lower():
                contbr_places.insert(i,places_check['Dominant Race'][j])
                break
            j=j+1
        if len(contbr_places)-1 < i:
            contbr_places.insert(i,np.nan)
        i=i+1
    
    return contbr_places

def get_candidate_data(year,state):
    i,j=0,0
    cand_party,cand_native,cand_work_state,cand_race= [],[],[],[]
    xlsfile = pd.ExcelFile(year+"\\"+"candidates"+"_"+year+".xlsx")
    candidates_data = xlsfile.parse('Sheet2')
    global raw_data
    
    while i<len(raw_data):
        j = 0
        while j<len(candidates_data):
            if raw_data['cand_nm'][i] == candidates_data['Name'][j]:
                cand_party.insert(i,candidates_data['Party'][j])
                cand_race.insert(i,candidates_data['Race'][j])
                if candidates_data['State of Origin'][j]==state:
                    cand_native.insert(i,1)
                else:
                    cand_native.insert(i,0)
                if candidates_data['Work State'][j]==state:
                    cand_work_state.insert(i,1)
                else: 
                    cand_work_state.insert(i,0)
                break;
            j=j+1
        i=i+1
    
    raw_data['cand_party']=cand_party
    raw_data['cand_native']=cand_native
    raw_data['cand_work_state']=cand_work_state
    raw_data['cand_race']=cand_race
    
def get_gender_data(social_data):
    i,j=0,0
    gender=[]
    global raw_data

    while i<len(raw_data):
        j=0
        while j<len(social_data):
            if social_data['City'][j].lower().find(raw_data['contbr_city'][i].lower())>-1:
                gender.insert(i,social_data['Dominant_gender'][j])
                break
            j=j+1
        if len(gender)-1 < i:
            gender.insert(i,np.nan)
        i=i+1
    
    return gender

def get_social_data(state,year):
    social_data = pd.read_csv(year+"\\"+"socialstats.csv",header=0,index_col=False)
    i=0
    social_city,dominant_gender=[],[]
    
    while i<len(social_data):
        social_city.insert(i,social_data['City'][i].split('>')[0])
        if social_data['Male'][i]>social_data['Female'][i]:
            dominant_gender.insert(i,1)
        else:
            dominant_gender.insert(i,0)
        i=i+1
    
    social_data['City']=social_city
    social_data['Dominant_gender']=dominant_gender
    
    social_data_state = social_data[social_data['State']==state]
    social_data_state = social_data_state.reset_index()
    return social_data_state

def get_city_labels(contbr_city,year):
    i=0
    urban_cities=[]
    f=open(year+"\\"+"Urban Areas2.txt")
    for line in f:
        urban_cities.insert(i,line.split(',')[0].split('--')[0])
        i = i+1
    i,j=0,0
    citylabel = [ ]
    #
    while i<len(contbr_city):
        j=0
        while j<len(urban_cities):
            if contbr_city[i].lower() == urban_cities[j].lower():
                citylabel.insert(i,1)
                break
            j=j+1
        if len(citylabel)-1 < i:
            citylabel.insert(i,0)
        i=i+1
        
    return citylabel

def get_candidate_race():
    global raw_data
    test1 = raw_data[['contbr_city_dominantrace']].T.to_dict().values()
    test2 = raw_data[['cand_race']].T.to_dict().values()
    
    vectorizer = DV(sparse=False)
    race_set1 = vectorizer.fit_transform(test1)
    race_set2 = vectorizer.fit_transform(test2)
    
    white1,black1,hisp1=[],[],[]
    white2,black2,hisp2=[],[],[]
    j,k=0,0
    
    for i in race_set1:
        temp = list(i)
        hisp1.insert(j,temp[0])
        black1.insert(j,temp[1])
        white1.insert(j,temp[2])
        j=j+1
        
    raw_data['city_White']=white1
    raw_data['city_Black']=black1
    raw_data['city_Hisp']=hisp1
    
    if race_set2.shape[1] == 3:
        for i in race_set2:
            temp = list(i)
            hisp2.insert(k,temp[0])
            black2.insert(k,temp[1])
            white2.insert(k,temp[2])
            k=k+1
    if race_set2.shape[1] == 2:
        for i in race_set2:
            temp = list(i)
            black2.insert(k,temp[0])
            white2.insert(k,temp[1])
            hisp2.insert(k,0)
            k=k+1
       
    raw_data['cand_White']=white2
    raw_data['cand_Black']=black2
    raw_data['cand_Hisp']=hisp2
    
def clean_data(year,state):
    urban_cities = []
    i=0  
    global raw_data
    test = pd.read_csv(year+"\\"+state+"_"+year+".csv",header=0,index_col=False)
    test = test[test.contb_receipt_amt>0]
    test = test[test.contbr_city.notnull()]
    test = test.reset_index()
    raw_data = test[0:10000]
    contbr_city = raw_data['contbr_city']
    #print len(contbr_city)
    citylabel = get_city_labels(contbr_city,year)
    raw_data['contbr_city_type'] = citylabel

    places_data = pd.read_csv(year+"\\"+"places.csv",header=0,index_col=False)
    social_data = get_social_data(state,year)
    
    gender=get_gender_data(social_data)
    raw_data['city_dom_gender']=gender
    
    contbr_places = get_race_data(places_data,state)
    raw_data['contbr_city_dominantrace']=contbr_places
    
    get_candidate_data(year,state)
    get_candidate_race()

def fit_classifier():

    global classifiers1
    global classifiers2
    global classifiers3
    
    zscore_amt = sc.zscore(raw_data['contb_receipt_amt'])
    raw_data['std_contb_amt'] = zscore_amt
    cleansed_data = raw_data[raw_data.contbr_city_dominantrace.notnull()]
    cleansed_data = cleansed_data[cleansed_data.contbr_occupation.notnull()]
    cleansed_data = cleansed_data[cleansed_data.city_dom_gender.notnull()]
    data_positivereceipts = cleansed_data[cleansed_data.contb_receipt_amt>0]

    #classifiers = [("Bagging(Tree)",BaggingRegressor(DecisionTreeRegressor()))]
    final_features = data_positivereceipts[['std_contb_amt','contbr_city_type','city_dom_gender','cand_native','cand_work_state','city_White','city_Black','city_Hisp','cand_White','cand_Black','cand_Hisp']]
    final_labels = data_positivereceipts['cand_party']
    
    classifiers1.fit(final_features,final_labels)
    classifiers2.fit(final_features,final_labels)
    classifiers3.fit(final_features,final_labels)
    
    return final_features,final_labels

def validate_classifier():
    
    global classifiers1
    global classifiers2
    global classifiers3
    
    zscore_amt = sc.zscore(raw_data['contb_receipt_amt'])
    raw_data['std_contb_amt'] = zscore_amt
    cleansed_data = raw_data[raw_data.contbr_city_dominantrace.notnull()]
    cleansed_data = cleansed_data[cleansed_data.contbr_occupation.notnull()]
    cleansed_data = cleansed_data[cleansed_data.city_dom_gender.notnull()]
    data_positivereceipts = cleansed_data[cleansed_data.contb_receipt_amt>0]
    
    final_features = data_positivereceipts[['std_contb_amt','contbr_city_type','city_dom_gender','cand_native','cand_work_state','city_White','city_Black','city_Hisp','cand_White','cand_Black','cand_Hisp']]
    final_labels = data_positivereceipts['cand_party']
    predictions1=classifiers1.predict(final_features)
    predictions2=classifiers2.predict(final_features)
    predictions3=classifiers3.predict(final_features)
    
    return final_labels,predictions1,predictions2,predictions3
states = ['FL','IN','OH','CO','NV','NH','IA']
for i in states:
    clean_data("2008",i)
    training_features,training_labels = fit_classifier()
    clean_data("2012",i)
    final_labels_2012,pred_dt,pred_svm,pred_knn = validate_classifier()
    cm1=confusion_matrix(final_labels_2012,pred_dt)
    cm2=confusion_matrix(final_labels_2012,pred_svm)
    cm3=confusion_matrix(final_labels_2012,pred_knn)
    print "**********FOR 2012 IN "+str(i)+"************"
    print cm1
    print cm2
    print cm3
    clean_data("2016",i)
    final_labels_2016,pred_dt,pred_svm,pred_knn = validate_classifier()
    cm1=confusion_matrix(final_labels_2016,pred_dt)
    cm2=confusion_matrix(final_labels_2016,pred_svm)
    cm3=confusion_matrix(final_labels_2016,pred_knn)
    print "***********FOR 2016 IN "+str(i)+"***********"
    print cm1
    print cm2
    print cm3

