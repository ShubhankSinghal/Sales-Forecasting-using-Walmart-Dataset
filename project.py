#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.special import boxcox1p
import seaborn as sns

#reading data from dataset(csv files)
features=pd.read_csv("features.csv")
stores=pd.read_csv("stores.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
#resetting train
train=train.groupby(['Store','Date'])['Weekly_Sales'].sum()
train=train.reset_index()
train.head(10)
#merging train and feature
data=pd.merge(train,features,on=['Store','Date'],how='inner')
data.head(10)
#marging store with data
data=pd.merge(data,stores,on=['Store'],how='inner')
data.head(10)
#sorting values of Data
data=data.sort_values(by='Date')
data.head(10)