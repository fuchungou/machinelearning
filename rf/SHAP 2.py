import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('../data/bandgap_no_duplicates.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']

output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[[output_param_name]].values
y = np.ravel(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(x_train, y_train)

explainer = shap.Explainer(model)

shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values, x_test, feature_names=input_param_name, plot_type="bar")
