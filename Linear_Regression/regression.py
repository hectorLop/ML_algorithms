from __future__ import annotations
from typing import Sequence

import numpy as np

class LinearRegression:
    """
    Implementation of Linear Regression algorithm with both Gradient Descent and
    Normal equation implementations.

    Parameters
    ----------
    gradient : bool
        Boolean that indicates to use gradient descent algorithm. Default value is True, 
        which implies the algorithm will use Gradient Descent
    learning_rate : float
        Determines the gradient step size. Default value is 0.01
    n_iterations : int
        Number of iterations of the Gradient Descent. Default value is 100.

    Attributes
    ----------
    _gradient : bool
        Boolean that indicates to use gradient descent algorithm
    _learning_rate : float
        Determines the gradient step size
    _n_iterations : int
        Number of iterations of the Gradient Descent
    _weights : array-like
        Weights vector
    """

    def __init__(self, gradient: bool=True, learning_rate: float=0.01, n_iterations: int=100):
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._gradient = gradient
        self._weights = None

    def _initialize_weights(self, n_features: int) -> None:
        """
        Initialize weigths values

        Parameters
        ----------
        n_features : int
            Number of features
        """
        self._weights = np.random.rand(n_features, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Find the optimal weights values for a given data

        Parameters
        ----------
        X : ndarray
            Training dataset
        y : ndarray
            Training target values dataset
        
        Returns
        -------
        LinearRegression
            Self object
        """
        self._check_matrix_dimensions(X, y)
      
        X = self._add_bias_to_data(X)
        self._initialize_weights(X.shape[1])

        # Normal equation if not gradient
        if not self._gradient:
            self._normal_equation(X, y)
        else:
            self._mse_gradient_descent(X, y)
        
        return self

    def _check_matrix_dimensions(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y number of rows must match in dimensions')
        elif y.shape[1] != 1:
            raise ValueError('Target values array must have an unique column')

    def _add_bias_to_data(self, X: np.ndarray) -> np.ndarray:
        """
        Adds the bias term to the data with a value of 1

        Parameters
        ----------
        X : ndarray
            Array of data

        Returns
        -------
        ndarray
            The data matrix with a column of ones appended to the beginning
        """
        bias_term = np.ones((X.shape[0], 1))

        return np.c_[bias_term, X]

    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Updates the weights vector using the Normal Equation.

        Parameters
        ----------
        X : ndarray
            Training data
        y : ndarray
            Training target values
        """
        inverse_term = np.linalg.inv(X.T.dot(X))
        second_term = X.T.dot(y)

        self._weights = inverse_term.dot(second_term)

    def _mse_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Computes the gradient descent for the MSE cost function on the weights
        parameter

        Parameters
        ----------
        X : ndarray
            Training data
        y : ndarray
            Training target values
        """
        m = X.shape[0]

        for iteration in range(self._n_iterations):
            error = X.dot(self._weights) - y
            gradient = 2/m * X.T.dot(error)
            self.weights = self._weights - self._learning_rate * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions from a given data

        Parameters
        ----------
        X : ndarray
            Data to make predictions

        Returns
        -------
        predictions : ndarray
            Predicted values
        """
        # Add a bias of 1 into the position 0 of X
        X = self._add_bias_to_data(X)

        predictions = X.dot(self._weights)

        return predictions
