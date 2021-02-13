from __future__ import annotations
from typing import Sequence
from abc import ABC, abstractmethod

import numpy as np

class Regression(ABC):

    def __init__(self, random_state: int=42) -> None:
        self._weights = None
        
        # Random seed for reproducible results
        if random_state:
            print(random_state)
            np.random.seed(random_state)

    def _initialize_weights(self, n_features: int):
        """
        Initialize weigths values

        Parameters
        ----------
        n_features : int
            Number of features

        Returns
        -------
        ndarray
            Array of weights initialized randomly """
        return np.random.randn(n_features, 1)

    def _check_matrix_dimensions(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Checks the input data dimmensions are right

        Parameters
        ----------
        X : ndarray
            Training data values
        y : ndarray
            Training data target values
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y number of rows must match in dimensions')
        elif len(y.shape) > 2 or (len(y.shape) == 2 and y.shape[1] != 1):
            raise ValueError(f'Target values array must have an unique column: {y.shape}')

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

    @abstractmethod
    def fit(self):
        pass

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


class LinearRegression(Regression):
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

    def __init__(self, random_state: int=42) -> None:
        super().__init__(random_state)

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
        self._weights = self._initialize_weights(X.shape[1])
     
        self._normal_equation(X, y)
        
        return self

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

class LinearRegressionGD(Regression):
    """
    Implementation of Linear Regression algorithm with both Gradient Descent and
    Normal equation implementations.

    Parameters
    ----------
    learning_rate : float
        Determines the gradient step size. Default value is 0.01
    n_iterations : int
        Number of iterations of the Gradient Descent. Default value is 100.
    batch_size : int
        Gradient Descent batch size. Default is 1, which means Stochastic Gradient
        Descent.

    Attributes
    ----------
    _learning_rate : float
        Determines the gradient step size
    _n_iterations : int
        Number of iterations of the Gradient Descent
    _weights : array-like
        Weights vector
    _batch_size : int
        Gradient Descent batch size
    """

    def __init__(self, learning_rate: float=0.1, n_iterations: int=100, batch_size: int=1,  random_state: int=42):
        super().__init__(random_state)
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._batch_size = batch_size

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
        self._weights = self._initialize_weights(X.shape[1])

        self._mse_gradient_descent(X, y)
        
        return self

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
        m = self._batch_size
        batches = m // self._batch_size

        for epoch in range(self._n_iterations):
            for iteration in range(0, m, self._batch_size):
                # Computing gradients
                error = X[iteration:iteration + self._batch_size].dot(self._weights) - y[iteration:iteration + self._batch_size]
                gradient = 2/m * X[iteration:iteration + self._batch_size].T.dot(error)

                # Updating weights
                self._weights = self._weights - self._learning_rate * gradient