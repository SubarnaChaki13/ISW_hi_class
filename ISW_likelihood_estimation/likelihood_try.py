from cobaya.likelihood import Likelihood
import numpy as np
import os

class MyLikelihood(Likelihood):
    """
    MyLikelihood is a class that extends the cobaya.likelihood.Likelihood class.
    It is designed to calculate the log-likelihood by comparing theoretical
    and observational data for ISW (Integrated Sachs-Wolfe).

    """

    def initialize(self):
        """
        Prepare any computation, importing any necessary code, files, etc.
        """
        self.data_cls = {}
        self.cov = {}
        self.w = {}
        self.inv_covs = {}

        # Load the data from the specified files
        k = 0
        for data in self.data_file:
            ISW_data = np.genfromtxt(data)[1, :]
            self.data_cls[k] = ISW_data
            k += 1

        # Load the cov files
        k = 0
        for cov in self.cov_file:
            ISWcov = np.genfromtxt(cov)
            self.cov[k] = ISWcov
            self.inv_covs[k] = np.linalg.inv(ISWcov)
            k += 1

        # Load the w files
        k = 0
        for w in self.w_file:
            ISW_w = np.genfromtxt(w)
            self.w[k] = ISW_w
            k += 1

    def get_requirements(self):
        """
        Return a dictionary specifying quantities calculated by a theory code are needed.
        """
        return {'ISW_cls': {'ISW': 6500}}

    def binning(self, cls_ub, ell, w):
        """
        Perform binning of theoretical C_l values.

        Args:
            cls_ub (array): Theoretical C_l values.
            ell (array): Multipole moments.
            w (array): Weighting factors.

        Returns:
            array, array: Binned C_l values and binned multipole moments.
        """
        cl_b = np.dot(w, cls_ub)
        ell_b = np.dot(w, ell)
        return cl_b, ell_b

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        Args:
            **params_values: Dictionary of sampled nuisance parameter values.

        Returns:
            float: Log-likelihood value.
        """
        theory_cls, den_ell = self.provider.get_ISW_cls()

        binned_cls = {}
        binned_ell = {}

        for key in theory_cls.keys():
            binned_cls[key], binned_ell[key] = self.binning(theory_cls[key], den_ell, self.w[key])

        theory_cls = binned_cls

        delta = {}
        chi2 = 0  # Initialize chi2 as 0

        for key in theory_cls.keys():
            delta[key] = self.data_cls[key] - theory_cls[key]
            chi2 += np.dot(delta[key], np.dot(self.inv_covs[key], delta[key]))  # Accumulate chi2 values

        return -chi2 / 2.
