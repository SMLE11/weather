# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 08:40:39 2021

@author: HUAWEI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

file_data="C:\\Users\\HUAWEI\\Desktop\\Prophet\\DataSet\\Hubei.csv"
CountryName="CN"
data_train=pd.read_csv(file_data)
length=len(data_train)
data_train_cut=data_train[30:length-1]
head=0
tail=29
len_data=len(data_train_cut)
 
for i in range(0,int(len_data/5)):
   m=Prophet(changepoint_prior_scale=2,n_changepoints=20     )
   m.add_country_holidays(country_name=CountryName)
   t=data_train[head:tail+1]
   m.fit(t)
   future=m.make_future_dataframe(5)
   forecast=m.predict(future)
   head+=5
   tail+=5
   if i==0:
       res=forecast[30:35]
   else:
       res=np.vstack((res,forecast[30:35]))
       
data_train_cut["ds"]=pd.to_datetime(data_train_cut["ds"]) 
len_res_column=res.shape[1]
len_res_row=res.shape[0]
val_theoretical =data_train_cut.loc[30:30+len_res_row-2,'y']
val_actual=res[0:len_res_row-1,len_res_column-1]

error=abs(val_theoretical-val_actual)
error_percentage=error/val_theoretical*100
ans=sum(error_percentage)/len(error_percentage)
print(ans)
