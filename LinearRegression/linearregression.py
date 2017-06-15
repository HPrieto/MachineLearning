from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, var, step=2, cor=False):
    """
            How Much:    How many datapoints to create
            Variance:    How variable do we want dataset
            Step:        How far on average to step on to yvalue
            Correlation: Pass value and want value to be neg, pos or none
    """
    val = 1
    ys = []
    # Create random yValues
    for i in range(hm):
        y = val + random.randrange(-var, var)
        ys.append(y)
        if cor and cor == 'pos':
            val += step
        elif cor and cor == 'neg':
            val -= step
    # Create random xvalues
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# plt.scatter(xs,ys)
# plt.show()


def best_fit_slope_and_intercept(xs, ys):
    """Find slope and y-intercept of data sets"""
    # slope
    m = (((mean(xs) * mean(ys)) - (mean(xs * ys))) /
         (mean(xs)**2 - mean(xs**2)))
    # y-intercept
    b = mean(ys) - m * mean(xs)
    return m, b


# Creates dataset
xs, ys = create_dataset(40, 10, 2, cor='pos')

# Get slope and y-intercept of dataset
m, b = best_fit_slope_and_intercept(xs, ys)

"""
For x in x's, multiply each dataset value by
the slope of the line and store values
"""
regression_line = [(m * x) + b for x in xs]

print('Slope:           ', m)
print('Intercept:       ', b)
print('Regression Line: ', regression_line)
"""
	More Linear Regression notes:
		error: Distance between each data point and best fit line
		squared error: e^2
			- we square error for penalizing outliers

		Coefficient of determination(r^2):
			SE == squared error
			YH == y-hatline == regression line == best fit line
			r^2 = (1 - ((SE(YH)) / (SE(mean(ys)))))
"""


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, ys_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


r_squared = coefficient_of_determination(ys, regression_line)

print('R^2(Accuracy): ', r_squared)

# Predict next value in 8
predict_x = 8

# Get y value of x_prediction using y = m x + b
predict_y = (m * predict_x) + b  # y=mx+b

plt.scatter(predict_x, predict_y, s=100, color='g')

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
