import pytest
import numpy as np
from sklearn import datasets

from Logistic_Regression.logistic_regression import LogisticRegression

@pytest.fixture
def train_test_data_final():
    iris = datasets.load_iris()

    X = iris['data']
    y = iris['target']

    return X, y

def test_logistic_regression(train_test_data_final):
    """
    Tests the linear regression algorithm using the Normal Equation
    """
    X_train, y_train = train_test_data_final
    X_train, y_train = X_train[:, 3:], (y_train == 2).astype(np.int8).reshape(-1, 1) # Binary classification problem
    X_test, y_test = np.array([[1.7], [1.5]]), np.array([[1], [0]])

    log_reg = LogisticRegression(n_iterations=5000, batch_size=32)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) > 0
    assert y_pred.shape[0] == X_test.shape[0]
    assert np.array_equal(y_test, y_pred)

def test_logistic_regression_softmax(train_test_data_final):
    X_train, y_train = train_test_data_final
    X_train = X_train[:, 2:]
    X_test, y_test = np.array([[5, 2]]), np.array([2])

    log_reg = LogisticRegression(n_iterations=5000, batch_size=32)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) > 0
    assert y_pred.shape[0] == X_test.shape[0]
    assert np.array_equal(y_test, y_pred)

def test_logistic_regression_softmax_loss(train_test_data_final):
    X_train, y_train = train_test_data_final
    X_train = X_train[:, 2:]

    log_reg = LogisticRegression(n_iterations=5000, batch_size=32)
    log_reg.fit(X_train, y_train)

    training_loss = log_reg._loss

    assert training_loss