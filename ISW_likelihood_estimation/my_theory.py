from cobaya.theory import Theory
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from math import pi
from classy import Class, CosmoComputationError
import matplotlib
import matplotlib.pyplot as plt

class CosmoClass(Theory):
    """
    CosmoClass extends cobaya.theory.Theory to interact with hi_class,
    implementing Horndeski's theory in the Cosmic Linear Anisotropy Solving System.
    It computes various linear cosmological observables, including
    FRW distances, CMB, matter power, and number count spectra.
    """

    def initialize(self):
        """
        Initialize method for setting up initial parameters.
        """
        try:
            # Load parameters from a file
            lna_smg, Delta_Mpl, Dkin, cs2 = np.loadtxt(self.designer_params, unpack=True)
        except Exception as e:
            print(f"Error reading parameters from file: {e}")
            raise

        # Generate arrays for dark energy parameters
        lna_de = np.linspace(-5., 0., endpoint=True, num=500)
        wext = -np.ones_like(lna_de)

        # Convert numpy arrays to strings for CLASS input
        self.hiclass_params['lna_smg'] = np.array2string(lna_smg, separator=',').replace('\n','').strip('[]')
        self.hiclass_params['Delta_M2'] = np.array2string(Delta_Mpl, separator=',').replace('\n','').strip('[]')
        self.hiclass_params['D_kin'] = np.array2string(Dkin, separator=',').replace('\n','').strip('[]')
        self.hiclass_params['cs2'] = np.array2string(cs2, separator=',').replace('\n','').strip('[]')
        self.hiclass_params['lna_de'] = np.array2string(lna_de, separator=',').replace('\n','').strip('[]')
        self.hiclass_params['de_evo'] = np.array2string(wext, separator=',').replace('\n','').strip('[]')
        self.state = {}

    def set(self, params_values_dict):
        """
        Set CLASS parameters using the provided parameter values.
        """
        self.cosmo = Class()
        self.cosmo.set(self.common_settings)
        self.cosmo.set(self.params)
        self.cosmo.set(self.standard_precision_params)
        self.cosmo.set(self.hiclass_params)
        self.cosmo.set(params_values_dict)

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Calculate cosmological observables based on the provided parameters.
        """
        self.set(params_values_dict)

        try:
            # Compute cosmological observables for the first set of spectra parameters
            self.cosmo.set(self.spectra_params1)
            self.cosmo.compute()
            # Calculate C_l values for the first set of spectra parameters
            cls1 = self.cosmo.raw_cl(6143)
            den_cls1 = self.cosmo.density_cl(6143)
            ell = cls1['ell']
            factor = 1.e10 * ell * (ell + 1.) / 2. / np.pi
            state['den_factor'] = 1.e8 * den_cls1['ell'] * (den_cls1['ell'] + 1.) / 2. / np.pi

            state['lensing_cls'] = {
                'den_ell': den_cls1['ell'],
                'den_cls': den_cls1['td']
            }
            self.cosmo.empty()

        except CosmoComputationError as e:
            print(f"Error computing results in the first compute: {e}")
            return -np.inf


        self.set(params_values_dict)

        try:
            # Compute cosmological observables for the second set of spectra parameters
            self.cosmo.set(self.spectra_params2)
            self.cosmo.compute()
            # Calculate C_l values for the second set of spectra parameters
            cls2 = self.cosmo.raw_cl(6143)
            den_cls2 = self.cosmo.density_cl(6143)

            state['density_cls'] = {
                'den_ell': den_cls2['ell'],
                'den_cls': den_cls2['td']
            }

        except CosmoComputationError as e:
            print(f"Error computing results in the second compute: {e}")
            return -np.inf


    def _get_ISW_cls(self, ell_factor=False, units="FIRASmuK2"):
        """
        Get the ISW (Integrated Sachs-Wolfe) C_l values.
        """
        den_ell = self.current_state['density_cls']['den_ell']
        ISW_cls = {}

        # Sum up the second item of both the states
        for k in self.current_state['density_cls']['den_cls'].keys():
            ISW_cls[k] = (self.current_state['density_cls']['den_cls'][k] + self.current_state['lensing_cls']['den_cls'][k])

        return ISW_cls, den_ell

    def get_ISW_cls(self, ell_factor=False, units="FIRASmuK2"):
        """
        Get the ISW (Integrated Sachs-Wolfe) C_l values.
        """
        return self._get_ISW_cls(ell_factor=ell_factor, units=units)
