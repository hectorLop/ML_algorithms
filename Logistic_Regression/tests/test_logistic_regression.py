import pytest
import numpy as np
from sklearn import datasets

from Logistic_Regression.logistic_regression import LogisticRegression

@pytest.fixture
def train_test_data():
    """
    Provides the training and test sets
    """
    iris = datasets.load_iris()

    X = iris['data'][:, 3:]

    y = (iris['target'] == 2).astype(np.int8)
    y = y.reshape(-1, 1)

    return X, y

@pytest.fixture
def train_test_data_softmax():
    """
    Provides the training and test sets
    """
    iris = datasets.load_iris()

    X = iris['data'][:, 2:]
    y = iris['target']

    return X, y

def test_logistic_regression(train_test_data):
    """
    Tests the linear regression algorithm using the Normal Equation
    """
    X_train, y_train = train_test_data
    X_test = np.array([[1.7], [1.5]])

    log_reg = LogisticRegression(n_iterations=5000, batch_size=32)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.size > 0
    assert y_pred.shape == (X_test.shape[0], 1)
    assert y_pred[0] == 1 
    assert y_pred[1] == 0

def test_logistic_regression_softmax(train_test_data_softmax):
    X_train, y_train = train_test_data_softmax
    X_test = np.array([[5, 2]])

    log_reg = LogisticRegression(n_iterations=5000, batch_size=32, softmax=True)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    
    assert y_pred[0] == 2