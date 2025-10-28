import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
with open('bike.csv','r',encoding='utf-8') as fp:
    bike_data=pd.read_csv(fp)
df=pd.DataFrame(bike_data)
df.drop(columns=['id'],inplace=True)
df.drop(df[df.city==0].index,inplace=True)
df.drop(columns=['city'],inplace=True)

df = df.reset_index()
df.drop(columns=['index'],inplace=True)

for i in range(4998):
    if (6<=df["hour"][i]<=18):
        df["hour"][i] =1
    else: df["hour"][i] =0
y_list=df["y"].to_numpy()

y_t=np.array([y_list]).T
df.drop(columns=['y'],inplace=True)
nmp=df.to_numpy()
x_train,x_test,y_train,y_test = train_test_split(nmp,y_t,test_size=0.2,shuffle=True)

min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)
y_train = min_max_scaler.fit_transform(y_train)
y_test = min_max_scaler.fit_transform(y_test)
model = LinearRegression()
model.fit(x_train,y_train)

y=model.predict(x_test)
print(mean_squared_error(y_test,y)**0.5)