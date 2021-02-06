import numpy as np

class LogisticRegression():

    def __init__(self, learning_rate: float=0.1, n_iterations: int=100, batch_size=1):
        self._weights = None
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._loss = []
        self._threshold = 0.5

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

    def fit(self, X, y):
        X = self._add_bias_to_data(X)
        self._weights = self._initialize_weights(X.shape[1])

        self._gradient_descent(X, y)

    def _gradient_descent(self, X, y):
        y = y.reshape(-1, 1)
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

                gradients = (1 / m) * batch.T.dot(y_pred - y_batches[idx])

                self._weights -= self._learning_rate * gradients
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X = self._add_bias_to_data(X)
        probs = self._sigmoid(X.dot(self._weights))

        predictions = (probs >= self._threshold).astype(int)

        return predictions