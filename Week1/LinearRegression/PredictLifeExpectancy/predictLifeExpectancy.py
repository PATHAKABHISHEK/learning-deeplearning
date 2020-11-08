from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

bmi_and_life_dataset = pd.read_csv("bmi_and_life_expectancy.csv")
x_values = bmi_and_life_dataset[["BMI"]]
y_values = bmi_and_life_dataset[["Life expectancy"]]

model = linear_model.LinearRegression()

model.fit(x_values, y_values)

predictied_life_epectancy = model.predict([[21.07931]])
plt.scatter(x_values, y_values)
plt.plot(x_values, model.predict(x_values))
plt.show()
print(predictied_life_epectancy)