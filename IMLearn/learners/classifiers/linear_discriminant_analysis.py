from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import IMLearn.metrics.loss_functions as loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        counts_dict = dict(zip(self.classes_, counts))

        self.mu_ = np.zeros((np.size(self.classes_), np.size(X, axis=1)))
        for k in range(self.classes_.size):
            mu_mle = np.zeros(np.size(X, axis=1))
            for i in range(np.size(X, axis=0)):
                if y[i] == k:
                    mu_mle += X[i]
            mu_mle /= counts_dict[k]
            self.mu_[k] = mu_mle

        self.cov_ = np.zeros((np.size(X, axis=1), np.size(X, axis=1)))
        for i in range(np.size(X, axis=0)):
            lhs = (X[i,:] - self.mu_[int(y[i]),:])
            lhs = lhs[:,None]
            self.cov_ += lhs @ lhs.T
        self.cov_ /= np.size(X, axis=0)

        self._cov_inv = inv(self.cov_)

        self.pi_ = np.zeros(self.classes_.size)
        for k in range(self.classes_.size):
            self.pi_[k] = counts_dict[k] / np.size(X, axis=0)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((np.size(X, axis=0), np.size(self.classes_)))

        for i in range(np.size(X, axis=0)):
            for k in range(self.classes_.size):
                posterior = np.log(self.pi_[k]) + np.matmul(np.matmul(X[i].T, self._cov_inv),
                                self.mu_[k]) - 0.5 * np.matmul(np.matmul(self.mu_[k].T, self._cov_inv), self.mu_[k])
                likelihood[i][k] = posterior

        return likelihood

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        return np.argmax(likelihood, axis=1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
<<<<<<< Updated upstream
        from ...metrics import misclassification_error
        raise NotImplementedError()
=======
        y_pred = self._predict(X)
        return loss_functions.misclassification_error(y_true=y, y_pred=y_pred)
>>>>>>> Stashed changes
