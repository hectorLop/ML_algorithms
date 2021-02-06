from __future__ import annotations

import numpy as np
from numpy import random
from Linear_Regression.regression import Regression

class LogisticRegression(Regression):

    def __init__(self, learning_rate: float=0.1, n_iterations: int=100, batch_size=1, random_state=42):
        super().__init__(random_state)
        self._weights = None
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._loss = []
        self._threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> Regression:
        """
        Find the optimal weight values.

        Parameters
        ----------
        X : array_like of shape {n_samples, n_features}
            Training vector
        y : array_like of shape {n_samples,}
            Target vector relative to X

        Returns
        -------
        self
            Fitted model
        """
        self._check_matrix_dimensions(X, y)

        X = self._add_bias_to_data(X)
        self._weights = self._initialize_weights(X.shape[1])

        self._gradient_descent_log_loss(X, y)

        return self

    def _gradient_descent_log_loss(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Apply gradient descent algorithm on log loss function.

        Parameters
        ----------
        X : array_like of shape {n_samples, n_features}
            Training vector
        y : array_like of shape {n_samples,}
            Target vector relative to X
        """
        # Computes the number of batches
        n_batches = X.shape[0] // self._batch_size

        # Splitting the data into batches
        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)

        for epoch in range(self._n_iterations):
            for idx, batch in enumerate(X_batches):
                m = len(batch)
                y_pred = self._sigmoid(batch.dot(self._weights))
            
                # Keep track of the loss for each epoch
                loss = (-1 / m) * np.sum((y_batches[idx] * np.log(y_pred)) + ((1 - y_batches[idx]) * np.log(1 - y_pred)))
                self._loss.append(loss)

                # Gradients and weights update
                gradients = (1 / m) * batch.T.dot(y_pred - y_batches[idx])
                self._weights -= self._learning_rate * gradients
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid function to given values.

        Parameters
        ----------
        z : float, integer or array_like
            Value to which apply the function
        
        Returns
        -------
        float or array_like
            New value
        """
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a given vector.

        Parameters
        ----------
        X : array_like, shape {n_samples, n_features}
            Samples vector
        
        Returns
        -------
        predictions : array, shape {n_samples,}
            Predicted class label per sample
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X = self._add_bias_to_data(X)
        probs = self._sigmoid(X.dot(self._weights))

        predictions = (probs >= self._threshold).astype(np.int8)

        return predictions