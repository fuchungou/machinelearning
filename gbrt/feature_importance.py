import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv('../data/averaged_bandgap_data.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[[output_param_name]].values
y = np.ravel(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model.fit(x_train, y_train)

explainer = shap.KernelExplainer(model.predict, x_train)

shap_values = explainer.shap_values(x_test)


mean_shap_values = np.mean(np.abs(shap_values), axis=0)

shap.summary_plot(shap_values, x_test, feature_names=input_param_name, plot_type="bar")


shap_summary_df = pd.DataFrame({'Feature': input_param_name, 'Mean_SHAP_Value': mean_shap_values})


shap_summary_df.to_csv('../data/feature_importance.csv', index=False)
