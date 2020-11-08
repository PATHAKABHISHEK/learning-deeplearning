from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

dataset = load_boston()

x_values = dataset["data"]
y_values = dataset["target"]

model = LinearRegression()

model.fit(x_values, y_values)

sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01, 6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02, 1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
prediction = model.predict(sample_house)
print("The Price of Sample House is {0}$.".format(prediction[0] * 1000))