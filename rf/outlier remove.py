import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


data = pd.read_csv('../data/averaged_bandgap_data.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[output_param_name].values

mae_results = []
r2_results = []

max_iterations = 10

removed_indices = []
removed_errors = []
removed_data = []

best_mae = float('inf')
best_model = None
best_x_train = None
best_y_train = None

for iteration in range(max_iterations):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=7)

    model = RandomForestRegressor(bootstrap=True, max_depth= 7, max_features= None, min_samples_leaf= 1, min_samples_split=3, n_estimators=32, random_state=42)
    model.fit(x_train, y_train.ravel())

    y_test_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    mae_results.append(mae)
    r2_results.append(r2)

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_x_train = x_train
        best_y_train = y_train

    y_train_pred = model.predict(x_train)
    train_errors = np.abs(y_train - y_train_pred)
    max_error_idx = np.argmax(train_errors)
    removed_indices.append(max_error_idx)
    removed_errors.append(train_errors[max_error_idx])
    removed_data.append((x[max_error_idx], y[max_error_idx], y_train_pred[max_error_idx]))
    x = np.delete(x, max_error_idx, axis=0)
    y = np.delete(y, max_error_idx)

    print(f"Iteration {iteration + 1}, MAE on test set: {mae}, R² on test set: {r2}")

y_test_pred_best = best_model.predict(x_test)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, max_iterations + 1), mae_results, marker='o', linestyle='-', color='blue')
plt.xlabel('Iteration')
plt.ylabel('MAE on Test Set')
plt.title('MAE on Test Set vs. Number of Iterations')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, max_iterations + 1), r2_results, marker='o', linestyle='-', color='green')
plt.xlabel('Iteration')
plt.ylabel('R² on Test Set')
plt.title('R² on Test Set vs. Number of Iterations')
plt.grid(True)

plt.tight_layout()
plt.show()

sorted_indices = np.argsort(y_test)
y_test_sorted = y_test[sorted_indices]
y_test_pred_sorted = y_test_pred_best[sorted_indices]

plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_sorted)), y_test_sorted, label='Actual Values', color='blue')
plt.plot(range(len(y_test_sorted)), y_test_pred_sorted, label='Predicted Values', color='red')
plt.xlabel('Index')
plt.ylabel('Target Value')
plt.title('Actual vs Predicted Values (Sorted)')
plt.legend()
plt.grid(True)
plt.show()