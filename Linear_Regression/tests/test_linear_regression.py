import pytest
import numpy as np

from regression import LinearRegression

def test_linear_regression_normal_equation():
    lin_reg = LinearRegression(False)
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 * X_train + np.random.randn(100, 1)

    X_test = np.array([[0], [2]])

    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred
    assert y_pred.shape == (X_test.shape[0], 1)

def test_linear_regression_grad():
    lin_reg = LinearRegression(True, 0.01)
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 * X_train + np.random.randn(100, 1)

    X_test = np.array([[0], [2]])

    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred
    assert y_pred.shape == (X_test.shape[0], 1)