import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('../data/averaged_bandgap_data.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']

output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[[output_param_name]].values
y = y.reshape(-1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)

rf_reg = RandomForestRegressor(bootstrap=True, max_depth= 7, max_features= None, min_samples_leaf= 1, min_samples_split=3, n_estimators=32, random_state=42)

rf_reg.fit(x_train, y_train)

y_pred = rf_reg.predict(x_test)
y_true = y_test

mse = mean_squared_error(y_true, y_pred)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)

r2 = r2_score(y_true, y_pred)
print("R2:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_reg.predict(x_test), color='blue', label='True vs Predicted')

max_val = max(np.max(y_test), np.max(rf_reg.predict(x_test)))
min_val = min(np.min(y_test), np.min(rf_reg.predict(x_test)))

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal line')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()
