import numpy as np

class Sigmoid():
    """
    Sigmoid activation function
    """
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the sigmoid function on a given vector

        Parameters
        ----------
        X : array_like, shape {n_samples, n_features}
            Input vector

        Returns
        -------
        array_like, shape {n_samples, n_features}
            Input vector transformed
        """
        return 1 / (1 + np.exp(-X))

class Softmax():
    """
    Softmax activation function
    """
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        """
        Performs the softmax function on a given vector

        Parameters
        ----------
        X : array_like, shape {n_samples, n_classes}
            Scores vector

        Returns
        -------
        array_like, shape {n_samples, n_classes}
            Scores vector transformed
        """
        exps = np.exp(scores)
        exps_sum = np.sum(exps, axis=1, keepdims=True)

        return exps / exps_sum  