import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('../data/averaged_bandgap_data.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[[output_param_name]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svr = SVR(kernel='rbf', C=78, epsilon=0.01, gamma='auto')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svr, x_train, y_train.ravel(), cv=kf, scoring='neg_mean_squared_error')

mse_scores = -scores
rmse_scores = np.sqrt(mse_scores)
mae_scores = -cross_val_score(svr, x_train, y_train.ravel(), cv=kf, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(svr, x_train, y_train.ravel(), cv=kf, scoring='r2')

print(f"MSE (5-fold CV): {np.mean(mse_scores):.4f} (± {np.std(mse_scores):.4f})")
print(f"RMSE (5-fold CV): {np.mean(rmse_scores):.4f} (± {np.std(rmse_scores):.4f})")
print(f"MAE (5-fold CV): {np.mean(mae_scores):.4f} (± {np.std(mae_scores):.4f})")
print(f"R2 (5-fold CV): {np.mean(r2_scores):.4f} (± {np.std(r2_scores):.4f})")

svr.fit(x_train, y_train.ravel())

y_pred = svr.predict(x_test)
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
plt.scatter(y_true, y_pred, color='blue', label='True vs Predicted')

max_val = max(np.max(y_true), np.max(y_pred))
min_val = min(np.min(y_true), np.min(y_pred))

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal line')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()