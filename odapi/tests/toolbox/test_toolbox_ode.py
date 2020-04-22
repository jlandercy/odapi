import sys
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from odapi.errors import BadParameter
from odapi.toolbox.ode import GGM, GRM

pd.options.display.max_rows = 100
pd.options.display.max_columns = 10


class ODETests:
    """Test GGM Model"""

    model = None
    params = None
    extra = {}
    atol = 1e-10
    show = False

    t = np.linspace(0, 20, 101)

    @classmethod
    def setUpClass(cls):
        cls.y = cls.model.ivp(cls.t, *cls.params)
        cls.sol = cls.model.regress(cls.t, cls.y, **cls.extra)

    def test_fit(self):
        if self.show:
            fig, axe = plt.subplots()
            axe.semilogy(self.t, self.y, 'o', label='Original')
            axe.semilogy(self.t, self.sol['yhat'], label='LM Fit')
            axe.legend()
            axe.grid(which='both', color='lightgray')
            plt.show()
        # print(self.y - self.sol['yhat'])
        self.assertTrue(np.allclose(self.y, self.sol['yhat'], self.atol))

    def test_params(self):
        # print(self.sol['popt'])
        self.assertTrue(np.allclose(self.params, self.sol['popt']))

    def test_covariance(self):
        # print(self.sol['pcov'])
        self.assertTrue(np.allclose(self.sol['pcov'], 0.))

    def test_test(self):
        test = self.model.test(self.sol['yhat'], self.y, ddof=len(self.sol['popt']))
        # print(test)
        self.assertTrue(all([np.isclose(v.pvalue, 1.) for v in test.values()]))


class GGMTests(ODETests, unittest.TestCase):

    model = GGM()
    params = (1., 2., 0.7)


class GRMTests(ODETests, unittest.TestCase):

    model = GRM()
    params = (1., 2., 0.7, 100, 1)
    extra = {'bounds': ([0.5, 1, 0.5, 50, 0.5], [1.5, 5, 1, 150, 1.5])}


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
