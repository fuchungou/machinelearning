import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


data = pd.read_csv('../data/averaged_bandgap_data.csv')


input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[[output_param_name]].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=29)

knn = KNeighborsRegressor(algorithm='auto', leaf_size= 25, metric='minkowski', n_neighbors=4, p=1, weights='distance')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
y_true = y_test

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

y_true_flat = y_true.flatten()
y_predictions_flat = y_pred.flatten()

print("均方误差(MSE):", mse)
print("均方根误差(RMSE):", rmse)
print("平均绝对误差(MAE):", mae)
print("决定系数(R2 Score):", r2)
