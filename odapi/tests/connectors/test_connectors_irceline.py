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


class DatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.client = Irceline()

    def test_dataset(self):
        meta = self.client.select(sitekey="41")
        records = self.client.get_records(meta, start="2019-01-01", stop="2022-03-01")
        data = records.merge(meta[["serieid", "seriekey"]])
        data = data.pivot_table(index="start", columns="seriekey", values="value")
        data.to_pickle("data.pickle", protocol=3)
        print(data)

    def test_conver(self):
        data = pd.read_pickle("data.pickle")
        data.to_pickle("data.pickle", protocol=0)

def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
