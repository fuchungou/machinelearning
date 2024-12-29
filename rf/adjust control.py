import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

data = pd.read_csv('../data/averaged_bandgap_data.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[output_param_name].values
y = y.reshape(-1)


initial_input = np.array([0.03, 0.7566, 0.2134, 1, 0, 0.146, 0.03, 0.824])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


rf_reg = RandomForestRegressor(bootstrap=True, max_depth= 40, max_features= None, min_samples_leaf= 1, min_samples_split=4, n_estimators=137, random_state=42)


rf_reg.fit(x_train, y_train)

def sensitivity_analysis(initial_input):

    def adjust_parameters(index, value):
        adjusted_input = initial_input.copy()

        if index == 0:
            diff = 0.03 - value
            adjusted_input[1] += diff * 0.7566/(0.7566+0.2134)
            adjusted_input[2] += diff * 0.2134/(0.7566+0.2134)
            adjusted_input[index] = value

        if index == 1:
            diff = 0.7566 - value
            adjusted_input[0] += diff * 0.03/(0.2134+0.03)
            adjusted_input[2] += diff * 0.2134/(0.2134+0.03)
            adjusted_input[index] = value
        if index == 2:
            diff = 0.2134 - value
            adjusted_input[0] += diff * 0.03 / (0.7566 + 0.03)
            adjusted_input[1] += diff * 0.7566 / (0.7566 + 0.03)
            adjusted_input[index] = value
        if index == 3:
            adjusted_input[4] = 1-value
            adjusted_input[index] = value
        if index == 4:
            adjusted_input[3] = 1-value
            adjusted_input[index] = value
        if index == 5:
            diff = 0.146 - value
            adjusted_input[6] += diff * 0.03 / (0.824 + 0.03)
            adjusted_input[7] += diff * 0.824 / (0.824 + 0.03)
            adjusted_input[index] = value
        if index == 6:
            diff = 0.03 - value
            adjusted_input[5] += diff * 0.146 / (0.146 + 0.824)
            adjusted_input[7] += diff * 0.824 / (0.146 + 0.824)
            adjusted_input[index] = value
        if index == 7:
            diff = 0.824 - value
            adjusted_input[5] += diff * 0.146 / (0.146 + 0.03)
            adjusted_input[6] += diff * 0.03 / (0.146 + 0.03)
            adjusted_input[index] = value

        return adjusted_input

    parameter_values = np.arange(0, 1.05, 0.05)

    for i in range(len(initial_input)):
        results = []
        for value in parameter_values:
            adjusted_input = adjust_parameters(i, value)
            prediction = rf_reg.predict([adjusted_input])
            results.append([value, prediction[0]])
            print(f"Adjusted Parameters: {adjusted_input}, Prediction: {prediction[0]}")

        df = pd.DataFrame(results, columns=[input_param_name[i], output_param_name])
        csv_path = os.path.join('../data', f'sensitivity_analysis_{input_param_name[i]}.csv')
        df.to_csv(csv_path, index=False)

        predictions = [rf_reg.predict([adjust_parameters(i, value)])[0] for value in parameter_values]
        plt.plot(parameter_values, predictions, marker='o')
        plt.title(f'Sensitivity Analysis for {input_param_name[i]}')
        plt.xlabel('Parameter Value')
        plt.ylabel(output_param_name)
        plt.grid(True)
        png_path = os.path.join('../data', f'sensitivity_analysis_{input_param_name[i]}.png')
        plt.savefig(png_path)
        plt.show()

sensitivity_analysis(initial_input)