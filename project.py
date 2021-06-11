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
sns.countplot(x="Type", data=data)

sns.boxplot(x='Type',y='Weekly_Sales',data=data)

data["Weekly_Sales"].plot.hist()

sns.countplot(x="IsHoliday", data=data)

data.isnull().sum()

sns.heatmap(data.isnull(),yticklabels=False, cmap="viridis")

data=data.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)
data.head(10)

data.isnull().sum()


sns.heatmap(data.isnull(),yticklabels=False, cmap="viridis")


data['Holiday']=[int(i) for i in list(data.IsHoliday)]
data.head(10)

Type_dummy=pd.get_dummies(data['Type'],drop_first=True)
Type_dummy.head(10)

data=pd.concat([data,Type_dummy],axis=1)
data.head(10)

data=data.drop(['Type','IsHoliday'],axis=1)
data.drop(10)

#splitting data in input and output
X=data.drop(['Weekly_Sales','Store','Date'],axis=1)
y=data['Weekly_Sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

LR=LinearRegression(normalize=True)
LR.fit(X_train,y_train)

y_pred=LR.predict(X_test)
plt.plot(y_test,y_pred,'ro')
plt.plot(y_test,y_test,'b-')
plt.show()

Root_mean_square_error=np.sqrt(np.mean(np.square(y_test-y_pred)))
print(Root_mean_square_error)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print(r2)

prediction=LR.predict(pd.DataFrame([(35.37,6.876,154.320056,10.964,503464,1,0,0)]))
print(prediction)

