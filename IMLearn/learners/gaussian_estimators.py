from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        '''

        TODO: which estimation to chose from ?
        :param X: ndarray of shape (n_samples, ), Training data
        :return: self : returns an instance of self.
        '''
        n = X.shape[0]
        self.mu_ = np.sum(X)/n
        self.var_ = np.var(X)
        self.fitted_ = True
        return self


    def pdf(self, X: np.ndarray) -> np.ndarray:
        '''
        Estimate Gaussian expectation and variance from given samples
        :param X:
        :return:
        '''
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        exponent = np.square(X - self.mu_) * -1/(2 * self.var_)
        scalar = 1/np.sqrt(2*np.pi*self.var_)
        pdf = scalar * np.exp(exponent)
        return pdf

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float: #TODO: I want to multiply the values in array returned by pdf yet the problem is that is received self.mu- and not mu
        '''
        :param mu: float , Expectation of Gaussian
        :param sigma: float, Variance of Gaussian
        :param X: ndarray of shape (n_samples, ) , Samples to calculate log-likelihood with
        :return: og_likelihood: float, log-likelihood calculated
        '''
        m= X.size
        var = np.square(sigma)
        exponent = ((X - mu)**2) * (-1 / (2 * var))
        scalar = 1 / np.sqrt(2 * np.pi * var)
        pdf = scalar * np.exp(exponent)
        likelihood = np.prod(pdf)
        return np.log(likelihood)




class MultivariateGaussian:

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features) #todo: the cov matrix is not n*n but n*n*(nc2)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features) #TODO who said that num of features = num samples !!!
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        n,m = X.shape
        # self.mu_ = np.sum(X,axis = 1) / n
        self.mu_ = np.mean(X, axis=1)
        self.cov_ = np.cov(X)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        '''
        Calculate PDF of observations under Gaussian model with fitted estimators
        :param X: ndarray (n_samples, n_features),    Samples to calculate PDF for
        :return:   pdfs: (n_samples, ) Calculated values of given samples for PDF function of N(mu_, cov_)

        '''
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        n = X.size
        det = np.linalg.det(self.cov_)
        inv = np.linalg.inv(self.cov_)
        exponent = np.exp((-1 / 2) * (X - self.mu_).T @ inv @ (X - self.mu_))
        scalar = 1 / ((2 * np.pi) ** n / 2) * det
        pdf = scalar * exponent
        return pdf
        #



    def log_likelihood(self,mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:

        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        inv_cov_ = np.linalg.inv(cov)
        m_observations= X.shape[0]
        n_features = X.shape[1] #shape of vector mu
        det = np.linalg.det(cov)
        arg1 = (m_observations/2)*np.log(abs(det))
        arg2 = (n_features * m_observations / 2) * np.log(2 * np.pi)
        arg3 = np.sum((1/2) * (X.T - mu) @ inv_cov_ @ (X.T - mu).T)
        likelihood = - arg1 - arg2 - arg3
        return likelihood

