import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


data = pd.read_csv('../data/bandgap_no_duplicates.csv')

input_param_name = ['MA', 'FA', 'Cs', 'Pb', 'Sn', 'Br', 'Cl', 'I']
output_param_name = 'Bandgap'

x = data[input_param_name].values
y = data[output_param_name].values
y = y.reshape(-1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVR(kernel='linear', C=100, epsilon=0.01, gamma='auto')

model.fit(x_train, y_train)

coefficients = model.coef_

coefficients = coefficients.flatten()

feature_importance_df = pd.DataFrame({'Feature Index': range(len(input_param_name)), 'Coefficient': coefficients})

feature_importance_df.to_csv('../data/feature_importance.csv', index=False)

plt.bar(range(len(coefficients)), coefficients)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient')
plt.title('Feature Importance')
plt.xticks(range(len(input_param_name)), input_param_name)  # 设置x轴刻度为特征名
plt.show()
