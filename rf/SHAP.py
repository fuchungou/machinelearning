import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap

data = pd.read_csv('../data/bandgap000.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[[output_param_name]].values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(x_test)


for i, feature_name in enumerate(input_param_name):

    feature_shap_values = shap_values[:, i]

    feature_values = x_test[:, i]
    df_feature_shap = pd.DataFrame({f'{feature_name}_value': feature_values, 'shap_value': feature_shap_values})

    df_feature_shap.to_csv(f'../data/{feature_name}_shap_values.csv', index=False)

    print(f"SHAP values for {feature_name} saved to '{feature_name}_shap_values.csv'")