import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
df=pd.read_csv("uber.csv")
df
df.head()
df.tail()
print(df.head())
df.isnull().sum()
df=df.drop(['Unnamed: 0','key'],axis=1)
df=df.dropna()
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df = df.drop('pickup_datetime', axis=1)
df=df[df['fare_amount'] > 0]
Q1=df['fare_amount'].quantile(0.25)
Q3=df['fare_amount'].quantile(0.75)
IQR=Q3-Q1
df=df[(df['fare_amount'] >= Q1-1.5*IQR) & (df['fare_amount'] <= Q3-1.5*IQR)]
sns.heatmap(df.corr(), annot=True)

plt.show()

X=df.drop('fare_amount',axis=1)
y=df['fare_amount']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)
P1=lr.predict(X_test)
print("linear Regression:",r2_score(y_test,P1) )
print("RMSE:",np.sqrt(mean_squared_error(y_test,P1)))

rf=RandomForestRegressor(n_estimators=10)
rf.fit(X_train,y_train)
P2=rf.predict(X_test)
print("Random Forest:",r2_score(y_test,P2))
print("RMSE:",np.sqrt(mean_squared_error(y_test,P2)))

print("linear Regression:",r2_score(y_test,P1) )
print("Random Forest:",r2_score(y_test,P2))

print("predicted fare:",rf.predict(X_test.iloc[[0]])[0])
