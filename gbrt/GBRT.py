import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv('../data/averaged_bandgap_data.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']

output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[output_param_name].values
y = y.reshape(-1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)


model = GradientBoostingRegressor(n_estimators=25, learning_rate=0.1, max_depth=6)


model.fit(x_train, y_train)

y_pred = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
r2 = r2_score(y_test, y_pred)
print("R2:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs Prediction')
plt.legend()
plt.show()
