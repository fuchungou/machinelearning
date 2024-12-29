import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_csv('../data/bandgap_no_duplicates.csv')

input_param_names = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_names].values
y = data[[output_param_name]].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


rf_reg = RandomForestRegressor(n_estimators=150, random_state=42)
rf_reg.fit(x_train, y_train.ravel())



def sensitivity_analysis(num_samples):
    results = []
    for _ in range(num_samples):

        p1 = np.random.rand()  # MA
        p2 = np.random.rand()  # FA
        p3 = np.random.rand()  # Cs


        p4 = 1  # Pb
        p5 = 0  # Sn
        p6 = 0.5  # Br
        p7 = 0.5  # Cl
        p8 = 0  # I


        prediction = rf_reg.predict([[p1, p2, p3, p4, p5, p6, p7, p8]])


        results.append((p1, p2, p3, prediction[0]))

    return results



num_samples = 1000
analysis_results = sensitivity_analysis(num_samples)


plt.figure(figsize=(10, 6))
plt.scatter(range(num_samples), [result[3] for result in analysis_results], alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Predicted Bandgap')
plt.title('Sensitivity Analysis of Randomly Assigned Parameters')
plt.show()
