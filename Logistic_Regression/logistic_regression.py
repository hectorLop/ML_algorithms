from __future__ import annotations
from typing import List, Tuple

import numpy as np
from numpy import random
from Linear_Regression.regression import Regression
from utils.activation_functions import Sigmoid, Softmax
from utils.encoders import OneHotEncoding

class LogisticRegression(Regression):

    def __init__(self, learning_rate: float=0.1, n_iterations: int=100, batch_size=1, random_state=42):
        super().__init__(random_state)
        self._weights = None
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._loss = []
        self._threshold = 0.5
        self._multiclass = False
        self._activation = Sigmoid()

    @property
    def loss(self):
        return self._loss

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
            Trained model
        """
        X = self._add_bias_to_data(X)
        self._determine_multiclass_or_binary(y) # Selects between multiclass and binary problems
        self._check_matrix_dimensions(X, y) # Check if the input dimensions are correct

        if self._multiclass:
            one_hot_encoding = OneHotEncoding()
            y = one_hot_encoding(y)
            self._weights = np.random.randn(X.shape[1], y.shape[1])
        else:
            self._weights = np.random.randn(X.shape[1], 1)

        self._gradient_descent_cross_entropy(X, y)

        return self

    def _determine_multiclass_or_binary(self, y: np.ndarray) -> None:
        """
        Determines if the target vector belongs to a binary or multiclass problem.

        Parameters
        ----------
        y : array_like, shape {n_samples, }
            Target vector
        """
        n_classes = len(np.unique(y))

        if n_classes > 2:
            self._multiclass = True
            self._activation = Softmax()

    def _gradient_descent_cross_entropy(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Apply gradient descent algorithm on cross entropy function.

        Parameters
        ----------
        X : array_like of shape {n_samples, n_features}
            Training vector
        y : array_like of shape {n_samples,}
            Target vector relative to X
        """
        X_batches, y_batches = self._compute_batches(X, y)

        for epoch in range(self._n_iterations):
            for idx, batch in enumerate(X_batches):
                m = len(batch)
                scores = batch.dot(self._weights)
                probs = self._activation(scores)

                # Keep track of the loss for each epoch
                loss = (-1 /m) * np.sum(np.sum(y_batches[idx] * np.log(probs + 1e-9)))
                self._loss.append(loss)

                # Gradients and weights update
                gradients = (1 / m) * batch.T.dot(probs - y_batches[idx])
                self._weights -= self._learning_rate * gradients

    def _compute_batches(self, X: np.ndarray, y: np.ndarray) -> Tuple[List, List]:
        """
        Computes the training batches.

        Parameters
        ----------
        X : array_like, shape {n_samples, n_features}
            Input data vector
        y : array_like, shape {n_samples, } or {n_samples, n_classes}
            Target vector relative to X

        Returns
        -------
        X_batches : List
            Input vector batches
        y_batches : List
            Target vector batches
        """
        n_batches = X.shape[0] // self._batch_size # Computes the number of batches

        X_batches = np.array_split(X, n_batches)
        y_batches = np.array_split(y, n_batches)

        return X_batches, y_batches        

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
            raise ValueError('Input vector must be an array')

        X = self._add_bias_to_data(X)
        probabilities = self._activation(X.dot(self._weights))

        if self._multiclass:
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = (probabilities >= self._threshold).astype(np.int8)

        return predictions