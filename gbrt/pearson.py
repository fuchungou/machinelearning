import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/averaged_bandgap_data.csv')


input_param_names = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I', 'Bandgap']


variables = [data[param].values for param in input_param_names]

means = np.mean(variables, axis=1)

diffs = [var - mean for var, mean in zip(variables, means)]

diff_square_sums = [np.sum(diff**2) for diff in diffs]

correlation_matrix = np.zeros((len(input_param_names), len(input_param_names)))

for i in range(len(input_param_names)):
    for j in range(len(input_param_names)):
        cov = np.sum(diffs[i] * diffs[j]) / len(data)
        std_i = np.sqrt(diff_square_sums[i] / len(data))
        std_j = np.sqrt(diff_square_sums[j] / len(data))
        correlation_matrix[i][j] = cov / (std_i * std_j)

print("Pearson 相关系数：")
for i, param_i in enumerate(input_param_names):
    for j, param_j in enumerate(input_param_names):
        print(f"{param_i} vs {param_j}: {correlation_matrix[i][j]}")

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')

plt.xticks(ticks=range(len(input_param_names)), labels=input_param_names, rotation=90)
plt.yticks(ticks=range(len(input_param_names)), labels=input_param_names)

plt.colorbar()

plt.title("Pearson Correlation Coefficient Matrix")
plt.xlabel("变量")
plt.ylabel("变量")

plt.show()
