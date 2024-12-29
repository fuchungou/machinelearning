import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


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

for iteration in range(max_iterations):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = KNeighborsRegressor(algorithm='auto', leaf_size=25, metric='minkowski', n_neighbors=4, p=1,
                              weights='distance')
    model.fit(x_train, y_train)


    y_test_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    mae_results.append(mae)
    r2_results.append(r2)

    y_train_pred = model.predict(x_train)
    train_errors = np.abs(y_train - y_train_pred)
    max_error_idx = np.argmax(train_errors)
    removed_indices.append(max_error_idx)
    removed_errors.append(train_errors[max_error_idx])
    removed_data.append((x[max_error_idx], y[max_error_idx], y_train_pred[max_error_idx]))
    x = np.delete(x, max_error_idx, axis=0)
    y = np.delete(y, max_error_idx)


    print(f"Iteration {iteration + 1}, MAE on test set: {mae}, R² on test set: {r2}")

print("Removed indices and corresponding errors:")
for i, (idx, error) in enumerate(zip(removed_indices, removed_errors)):
    print(f"Iteration {i + 1}: Index {idx}, Error {error}")


print("\nRemoved data points including input parameters, actual output, and predicted values:")
for i, (input_params, actual_output, predicted_output) in enumerate(removed_data):
    print(
        f"Iteration {i + 1}: Input {input_params}, Actual Output {actual_output}, Predicted Output {predicted_output}")


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