import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from odapi.errors import BadParameter
from odapi.toolbox.ode import GGM, GRM

pd.options.display.max_rows = 100
pd.options.display.max_columns = 10


class ODETests:
    """Test GGM Model"""

    model = None
    params = None
    t = np.linspace(0, 20, 100)

    def test_fit(self):
        y = self.model.ivp(self.t, *self.params)
        yh = self.model.regress(self.t, y)
        print(y, yh)


class GGMTests(ODETests, unittest.TestCase):

    model = GGM()
    params = (1., 2., 0.7)


class GRMTests(ODETests, unittest.TestCase):

    model = GRM()
    params = (1., 2., 0.7, 100, 1)


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
