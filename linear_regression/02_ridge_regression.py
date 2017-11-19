import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

"""
Linear regression performed on the dataset food_truck_profit.csv
The first column is the population of a city and the second column is the profit of a food truck in that city.
A negative value for profit indicates a loss. Values are given in 10,000s of dollars.
"""


def show_linear_regression_model_as_str(function_name, w1, w0=None):
    """
    function_name = f
    w0 = 4
    w0 = 1.5
    Function will return "f_w(x) = 4 + 1.5 * x"
    :param function_name: Name of the regression function (for example f)
    :param w1: Constant which will multiply x
    :param w0: Constant that will be added (intercept)
    :return:
    """
    if w0:
        return "%s_w(x) = %f + %f * x" % (function_name, w0, w1)
    else:
        return "%s_w(x) = %f * x" % (function_name, w1)


def main():
    # We load our data
    df = pd.read_csv('../datasets/food_truck_profit.csv')

    print(df.columns.tolist())

    x = df[['population']]
    y = df[['profit']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

    # Try changing alpha through values [0.1, 1, 100, 1000, 1500, 2000, 10 000, 100 000
    lr = Ridge(alpha=0.1)
    lr.fit(x_train, y_train)
    print(show_linear_regression_model_as_str("f", w1=lr.coef_[0][0], w0=lr.intercept_))

    train_prediction = lr.predict(x_train)
    test_prediction = lr.predict(x_test)

    mse_train = mean_squared_error(y_train, train_prediction)
    mse_test = mean_squared_error(y_test, test_prediction)
    r2_train = r2_score(y_train, train_prediction)
    r2_test = r2_score(y_test, test_prediction)

    print("MSE training: ", mse_train)
    print("MSE test: ", mse_test)
    print("R^2 training: ", r2_train)
    print("R^2 test: ", r2_train)

    # Draw our model prediction
    plt.scatter(x_train, y_train, color='red', marker='x', label='Training set')
    plt.plot(x_train, train_prediction, color='red', label='Training set, $R^2$=%.2f' % r2_train)
    plt.scatter(x_test, y_test, color='blue', label='Test set')
    plt.plot(x_test, test_prediction, color='blue', label='Test set, $R^2$=%.2f' % r2_test)
    plt.xlabel('population* $10^4$')
    plt.ylabel('profit*$10^4\$$')
    plt.legend(loc='lower right')
    plt.title('Ridge linear regression')
    plt.show()


if __name__ == "__main__":
    main()
