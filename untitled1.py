# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 08:51:41 2021

@author: HUAWEI
"""

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
CountryName="UK"
data_train=pd.read_csv(file_data)
length=len(data_train)
data_train_cut=data_train[30:length-1]
array1=data_train_cut.loc[30:30+60-1,'y']