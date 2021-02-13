import numpy as np

class OneHotEncoding():
    def __call__(self, data):
        """
        Perform OneHotEncoding on the target vector

        Parameters
        ----------
        y : array_like, shape {n_samples, }
            Targets vector
        
        Returns
        -------
        y_one_hot : array_like, shape {n_samples, n_classes}
            Target vector encoded
        """
        n_categories = len(np.unique(data))
        n_samples = data.shape[0]

        y_one_hot = np.zeros((n_samples, n_categories))
        y_one_hot[np.arange(n_samples), data] = 1 # Access all training instances and y-value positions

        return y_one_hot