import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
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

    # Load our dataset
    # x = df['population']
    # y = df['profit']
    # x = x.values.reshape((-1, 1))
    # y = y.values.reshape((-1, 1))

    # or like this
    x = df[['population']]
    y = df[['profit']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

    # ------------------------------------------------------------------------------------------------------------------
    # We construct a linear regression model and perform the training process.
    # scikit-learn will use an optimization algorithm to minimize the loss function and give us
    # weights w_i that minimize the loss function.
    # ------------------------------------------------------------------------------------------------------------------
    # Perform training
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    # Show generated model
    print(show_linear_regression_model_as_str("f", w1=linear_regression.coef_[0][0], w0=linear_regression.intercept_[0]))

    # Calculate predictions on the training and train set
    y_train_pred = linear_regression.predict(x_train)
    y_test_pred = linear_regression.predict(x_test)

    # Calculate the mean squared error
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculated r squared
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("MSE training: ", mse_train)
    print("MSE test: ", mse_test)
    print("R^2 training: ", r2_train)
    print("R^2 test: ", r2_test)
    print()

    # ------------------------------------------------------------------------------------------------------------------
    #  We can also construct the model without using w_0
    # Setting parameter fit_intercept to false constructs a LR object that has w_0 = 0
    # ------------------------------------------------------------------------------------------------------------------
    lr2 = LinearRegression(fit_intercept=False)
    lr2.fit(x_train, y_train)

    print(show_linear_regression_model_as_str("g", w1=linear_regression.coef_[0][0]))

    train_pred2 = lr2.predict(x_train)
    test_pred2 = lr2.predict(x_test)

    mse_train2 = mean_squared_error(y_train, train_pred2)
    mse_test2 = mean_squared_error(y_test, test_pred2)

    r2_train2 = r2_score(y_train, train_pred2)
    r2_test2 = r2_score(y_test, test_pred2)

    print("MSE training: ", mse_train2)
    print("MSE test: ", mse_test2)
    print("R^2 training: ", r2_train2)
    print("R^2 test: ", r2_test2)

    plt.scatter(x_test, y_test, c='k', s=10, label='Test set')
    plt.plot(x_test, y_test_pred, c='r',
             label='Function f: MSE=%.2f, $R^2$=%.2f' % (mse_test, r2_test))
    plt.plot(x_test, test_pred2, c='b',
             label='Function g: MSE=%.2f, $R^2$=%.2f' % (mse_test2, r2_test2))
    plt.xlabel('population*$10^4$')
    plt.ylabel('profit*$10^4$')
    plt.legend(loc='upper left', fontsize=6)
    plt.title('Introduction to Linear regression')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
