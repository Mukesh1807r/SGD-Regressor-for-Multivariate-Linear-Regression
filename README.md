# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Mukesh R
RegisterNumber: 212224240098  
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
print(data)

import pandas as pd
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

print(df.tail())

print(df.info())

x=df.drop(columns=['AveOccup','target'])
x.info()
print(x.shape)

y=df[['AveOccup','target']]
y.info()
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
x.head()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)

y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

print("\nPredictions:\n",y_pred[:5])
```

## Output:

![image](https://github.com/user-attachments/assets/47807dca-7e1c-4ee6-a730-4d9e605595f5)

![image](https://github.com/user-attachments/assets/348cf98d-8caf-4f36-9766-83d8fbc3d6dd)

![image](https://github.com/user-attachments/assets/367cfcd1-f8e8-4b02-9b70-58088f50d395)

![image](https://github.com/user-attachments/assets/f2a2fd57-7faa-4b31-88aa-bdd00781270d)

![image](https://github.com/user-attachments/assets/1b9694ed-3ef0-4a9a-b700-5839bf6bba71)

![image](https://github.com/user-attachments/assets/9e082f30-eacf-4691-852d-adc2809aa36e)

![image](https://github.com/user-attachments/assets/e66287c8-739e-49e6-873b-07e266639b25)

![image](https://github.com/user-attachments/assets/70cda3b8-4803-4455-abc5-8e58f1bc9a28)

![image](https://github.com/user-attachments/assets/f5476cca-1c81-456f-8d20-bcf89738b88e)

![image](https://github.com/user-attachments/assets/eab90bbc-8473-4ad3-956b-fd9b7f726c60)

![image](https://github.com/user-attachments/assets/a7e4d65e-3018-43a6-802c-055bb63e881b)

![image](https://github.com/user-attachments/assets/836cb912-a5de-47c4-bc99-050acb509040)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
