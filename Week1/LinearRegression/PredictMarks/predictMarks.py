# Predicting Student Marks Based on Number of Hours Student Studied

# Here We will Be doing Linear Regression and Gradient Descent By Our Own

from numpy import *

def step_descent(b, m, points, learning_rate):
    gradient_b = 0
    gradient_m = 0
    totalError = 0
    N = len(points)
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        gradient_b += -(2/N) * (y - ((m * x) + b)) 
        gradient_m += -(2/N) * x * (y - ((m * x) + b)) 
        totalError += (y - ((m  * x) + b)) ** 2
    new_b = b - (learning_rate * gradient_b)
    new_m = m - (learning_rate * gradient_m)
    totalError = totalError / len(points)
    return [new_b, new_m, totalError]

def gradient_descent(initial_b, initial_m, points, iterations, learning_rate):
    b = initial_b
    m = initial_m

    for i in range(0, iterations):
        [b, m, totalError] = step_descent(b, m, points, learning_rate)
    return [b, m, totalError]

def run():
    points = array(genfromtxt("data.csv", delimiter=","))
    
    # y = mx + b slope-intercept formula
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0

    iterations = 1000
    
    print("Running.....")
    [b, m, totalError] = gradient_descent(initial_b, initial_m, points, iterations, learning_rate)
    print("After {0} iterations b = {1} , m = {2} and error = {3}".format(1000, b, m, totalError))


if __name__ == '__main__':
    run()