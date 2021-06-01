import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.metrics import confusion_matrix,precision_score,recall_score,precision_recall_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#import xgboost as xgb
#import shap
import csv

#df = pd.read_csv('/kaggle/input/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv')
filepath = '../data/lending-club-loan-data/loan.csv'
df = pd.read_csv('../data/lending-club-loan-data/loan.csv')
data = df.copy()

columns = data.columns
for i in range (len(columns)):
    print (data.loc[:,columns[i]].dtype)
data = data.drop(['id','policy_code','out_prncp','out_prncp_inv','url','pymnt_plan','hardship_flag','grade'], axis=1)
#((df.isnull().sum())/len(df)*100).plot.bar(title='Percentage of missing values per column')
MAX_COL_PERC = 0.02

perc = data.isnull().sum() / len(df)     # .sort_values(ascending=False)
na_cols = perc.iloc[np.where(np.array(perc)>MAX_COL_PERC)].index
print(len(na_cols), "columns dropped.")
# Uncomment to show which columns are dropped
# with pd.option_context('display.max_rows',None):
#     display(df.loc[:,na_cols].describe().transpose())
data = data.drop(na_cols, axis=1)
#sn.countplot(y='loan_status', data=data)
data = data[(data['loan_status'] == 'Fully Paid') | (data['loan_status'] == 'Current')|(data['loan_status'] == 'Charged Off')]
data['label'] = data.apply(lambda r: 2 if r['loan_status'] == 'Fully Paid' else 1 if r['loan_status'] == 'Current' else 0, axis=1)
data = data.drop('loan_status', axis=1)
date_fields = ['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
for col in date_fields:  
    data[col] = pd.to_datetime(data[col]) 
    data[col + '_month'] = data[col].dt.month
    data[col + '_year'] = data[col].dt.year
#pd.to_datetime(df.astype(str).apply('-'.join, 1))
data = data.drop(date_fields, axis=1)
MIN_DUMMY_PERC = 0.01

vc = data['title'].value_counts()
titles = vc.iloc[np.where(np.array(vc)>MIN_DUMMY_PERC*len(df))].index
data['title'] = data.apply(lambda r: r['title'] if r['title'] in titles else 'Other title',axis=1)
#for i in range (len(columns)):
    #if (df.loc[:, columns[i]].dtype == 'object') and (columns[i] != 'addr_state'):
addresses = data['addr_state'].tolist()
#loan_status = data['loan_status'].to_list()
cat = data.select_dtypes(include=['object']).columns 
data[cat] = data[cat].fillna(value='Missing')
#df_cat = pd.get_dummies(data=df.loc[:,~df.columns.isin(['addr_state'])])
data_cat = pd.get_dummies(data=data)

print("Total # categorical columns: ", len(data_cat.columns))
data_cat = data_cat.drop([col for col, cnt in data_cat.sum().iteritems() if cnt < MIN_DUMMY_PERC*len(data_cat)], axis=1)
print("Reduced # categorical columns: ", len(data_cat.columns))

data = data.drop(cat,axis=1)
data = pd.concat([data,data_cat], axis=1)
data['addr_state'] = addresses
#data['loan_status'] = loan_status
#data=data.drop('nan', axis=1)
data=data.fillna(0)
list_obj = []
list_1 = []
list_10 = []
list_100 = []
list_10000 = []
columns = data.columns
df = data.copy()
df = df.loc[:,~df.columns.duplicated()]
print('columns', columns)
for i in range(len(columns)):
    print (df.loc[:,columns[i]])
    if (columns[i] == 'nan'):
        df = df.drop('nan', axis = 1)
    elif(columns[i] == '0'):
        df = df.drop('0', axis = 1)

    elif ((df.loc[:,columns[i]].dtype == object) and (columns[i] != 'addr_state')):
    #if(df.select_dtypes(include=['object']) and (columns[i] != 'addr_state')):
        list_obj.append(columns[i])
        value = list(df.drop_duplicates(columns[i]).loc[:, columns[i]])
        print(value)
        for j in range(len(value)):
            df.loc[df[columns[i]] == value[j], columns[i]] = j
    elif ((df.loc[:, columns[i]].dtype == 'float64') or (df.loc[:, columns[i]].dtype == 'int64')):
        print(columns[i])
        if (df[columns[i]].mean() > 10.0) and (df[columns[i]].mean() <= 100.0):
            list_10.append(columns[i])
            df[columns[i]] = df[columns[i]] / 10
        elif (df[columns[i]].mean() > 100.0) and (df[columns[i]].mean() <= 1000.0):
            list_100.append(columns[i])
            df[columns[i]] = df[columns[i]] / 100
        elif (df[columns[i]].mean() > 1000.0):
            list_10000.append(columns[i])
            df[columns[i]] = df[columns[i]] / 10000
        elif (df[columns[i]].mean() <= 10.0):
            list_1.append(columns[i])
state_set = list(set(df.loc[:,'addr_state']))
print('state_set', state_set)
#state_set = state_set.remove('0')

save_dirs = '../data/loan/'

import os
if not os.path.exists(save_dirs):
    os.makedirs(save_dirs)

for j in range(len(state_set)):
    if (columns[i] == 'addr_state_'+str(state_set[j])):
        df = df.drop('addr_state_'+str(state_set[j]), axis =1)
for j in range(len(state_set)):
    print('saving: ', state_set[j])
    data_new = df.loc[df['addr_state']== state_set[j]].drop(['addr_state'], axis=1)
    #data_new = data_new.drop('addr_state_'+str(state_set[j]), axis =1)
    #with open(save_dirs+'/loan_'+ str(state_set[j]) + '.csv', 'w', newline='',encoding='utf-8') as csv_file:
        #csv_writer = csv.writer(csv_file)
        #csv_writer.writerow(data_new.columns)
        #for i in range(data_new.shape[0]):
            #csv_writer.writerow(data_new.iloc[[i][0]])
    data_new.to_csv(save_dirs+'/loan_'+ str(state_set[j])+'.csv', index =False)

print('done')
