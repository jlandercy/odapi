import sys

import unittest

import pandas as pd

from odapi.tests.interfaces.test_interfaces_timeserie import TimeSerieAPITest
from odapi.connectors.opendata import Irceline


class IrcelineTests(TimeSerieAPITest, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup Meta and Data Samples for testing"""
        cls.client = Irceline()
        cls.channel_sample = {'10607', '6574', '6621', '99917'}
        cls.freq = pd.Timedelta('1H')
        super().setUpClass()


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
