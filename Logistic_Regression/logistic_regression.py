from __future__ import annotations

import numpy as np
from numpy import random
from Linear_Regression.regression import Regression

class LogisticRegression(Regression):

    def __init__(self, learning_rate: float=0.1, n_iterations: int=100, batch_size=1, random_state=42, softmax=False):
        super().__init__(random_state)
        self._weights = None
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._loss = []
        self._threshold = 0.5
        self._softmax = softmax

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
        X = self._add_bias_to_data(X)

        if self._softmax:
            y = self._to_one_hot(y)
            self._weights = self._initialize_multiclass_weights(X, y)
            self._gradient_descent_cross_entropy(X, y)
        else:
            self._check_matrix_dimensions(X, y)

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

    def _gradient_descent_cross_entropy(self, X: np.ndarray, y: np.ndarray):
        # Computes the number of batches
        n_batches = X.shape[0] // self._batch_size

        # Splitting the data into batches
        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)

        for epoch in range(self._n_iterations):
            for idx, batch in enumerate(X_batches):
                m = len(batch)
                scores = batch.dot(self._weights)
                probs = self._softmax_function(scores)

                gradients = (1 / m) * batch.T.dot(probs - y_batches[idx])
                self._weights -= self._learning_rate * gradients

    def _initialize_multiclass_weights(self, X, y):
        n_features = X.shape[1]
        n_classes = y.shape[1]

        return np.random.randn(n_features, n_classes)

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

    def _softmax_function(self, scores: np.ndarray) -> np.ndarray:
        exps = np.exp(scores)
        exps_sum = np.sum(exps, axis=1, keepdims=True)

        return exps / exps_sum

    def _to_one_hot(self, y):
        n_classes = len(np.unique(y))
        m = y.shape[0]

        y_one_hot = np.zeros((m, n_classes))
        # Access all training instances and y-value positions
        y_one_hot[np.arange(m), y] = 1

        return y_one_hot

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

        if self._softmax:
            probs = self._softmax_function(X.dot(self._weights))
            predictions = np.argmax(probs, axis=1)

            return predictions
        else:
            probs = self._sigmoid(X.dot(self._weights))

            predictions = (probs >= self._threshold).astype(np.int8)

            return predictions