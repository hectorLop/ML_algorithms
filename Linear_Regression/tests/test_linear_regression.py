import pytest
import numpy as np

from Linear_Regression.regression import LinearRegression, LinearRegressionGD

@pytest.fixture
def train_test_data():
    """
    Provides the training and test sets
    """
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 * X_train + np.random.randn(100, 1)
    X_test = np.array([[0], [2]])

    return X_train, y_train, X_test


def test_linear_regression_normal_equation(train_test_data):
    """
    Tests the linear regression algorithm using the Normal Equation
    """
    lin_reg = LinearRegression()
    X_train, y_train, X_test = train_test_data

    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.size > 0
    assert y_pred.shape == (X_test.shape[0], 1)
    assert y_pred[0] < y_pred[1]

def test_linear_regression_batch_gradient(train_test_data):
    """
    Tests the linear regression algorithm using Batch Gradient Descent
    """
    X_train, y_train, X_test = train_test_data
    lin_reg = LinearRegressionGD(batch_size=X_train.shape[0], n_iterations=10000)

    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.size > 0
    assert y_pred.shape == (X_test.shape[0], 1)
    assert y_pred[0] < y_pred[1]

def test_linear_regression_stochastic_gradient_descent(train_test_data):
    lin_reg = LinearRegressionGD(n_iterations=1000)
    X_train, y_train, X_test = train_test_data

    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    y_pred = lin_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.size > 0
    assert y_pred.shape == (X_test.shape[0], 1)
    assert y_pred[0] < y_pred[1]

def test_linear_regression_mini_batch_gradient_descent(train_test_data):
    lin_reg = LinearRegressionGD(batch_size=6, n_iterations=5000)
    X_train, y_train, X_test = train_test_data

    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    y_pred = lin_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.size > 0
    assert y_pred.shape == (X_test.shape[0], 1)
    assert y_pred[0] < y_pred[1]

def test_linear_regression_different_dimensions(train_test_data):
    """
    Test the exception when both arrays have different dimensions 
    """
    lin_reg = LinearRegression()
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 + np.random.randn(105, 1)

    try:
        lin_reg.fit(X_train, y_train)
        assert False
    except ValueError as ex:
        assert str(ex) == 'X and y number of rows must match in dimensions'

def test_linear_regression_multiple_target_columns(train_test_data):
    """
    Test the exception when both the target data has more than one column.
    """
    lin_reg = LinearRegression()
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 + np.random.randn(100, 2)

    try:
        lin_reg.fit(X_train, y_train)
        assert False
    except ValueError as ex:
        assert str(ex) == 'Target values array must have an unique column'