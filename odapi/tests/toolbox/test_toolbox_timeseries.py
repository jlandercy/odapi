import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from odapi.errors import BadParameter
from odapi.toolbox.timeseries import TimeSeries

pd.options.display.max_rows = 100
pd.options.display.max_columns = 10


class TimeGrouperTest(unittest.TestCase):
    """Test Time Grouper function"""

    def setUp(self):
        self.mon = pd.Timestamp("2020-04-13 12:00:00", tz="CET")
        self.sun = pd.Timestamp("2020-04-19 12:00:00", tz="CET")
        self.week = pd.date_range(self.mon, self.sun, freq='1D')

    def test_weekly_dayofweek(self):
        """Check day of week is correct when mapped onto generic week"""
        self.assertTrue(all(self.week.dayofweek == self.week.map(TimeSeries.weekly).dayofweek))


class TimePerformanceTest(unittest.TestCase):
    """Test Time Performance function"""

    def setUp(self):
        self.t = pd.date_range('2020-01-01', '2021-01-01', freq='1H', closed='left')
        self.dataset = pd.DataFrame(np.random.randn(self.t.size, 9), index=self.t)
        self.dataset.loc[:'2020-01-01', 1] = np.nan
        self.dataset.loc['2020-01-02':, 2] = np.nan
        self.dataset.loc[:'2020-12-30', 3] = np.nan
        self.dataset.loc['2020-12-31':, 4] = np.nan
        self.dataset.loc['2020-04-19':'2020-04-19', 5] = np.nan
        self.dataset.loc[:'2020-04-18', 6] = np.nan
        self.dataset.loc['2020-04-20':, 6] = np.nan
        self.dataset.loc['2020-02-29':'2020-02-29', 7] = np.nan
        self.dataset.loc[:'2020-02-28', 8] = np.nan
        self.dataset.loc['2020-03-01':, 8] = np.nan

    def test_performance_inside(self):
        n = 24*366
        perf = self.dataset.apply(TimeSeries.performance, limit_area='inside')
        self.assertEqual(self.dataset.shape[0], n)
        self.assertEqual(perf.loc['expected_count', 0], n)
        self.assertEqual(perf.loc['real_count', 0], n)
        self.assertAlmostEqual(perf.loc['performance', 0], 1.)
        self.assertEqual(perf.loc['expected_count', 1], n-24)
        self.assertEqual(perf.loc['real_count', 1], n-24)
        self.assertAlmostEqual(perf.loc['performance', 1], 1.)
        self.assertEqual(perf.loc['expected_count', 2], 24)
        self.assertEqual(perf.loc['real_count', 2], 24)
        self.assertAlmostEqual(perf.loc['performance', 2], 1.)
        self.assertEqual(perf.loc['expected_count', 3], 24)
        self.assertEqual(perf.loc['real_count', 3], 24)
        self.assertAlmostEqual(perf.loc['performance', 3], 1.)
        self.assertEqual(perf.loc['expected_count', 4], n-24)
        self.assertEqual(perf.loc['real_count', 4], n-24)
        self.assertAlmostEqual(perf.loc['performance', 4], 1.)
        self.assertEqual(perf.loc['expected_count', 5], n)
        self.assertEqual(perf.loc['real_count', 5], n-24)
        self.assertAlmostEqual(perf.loc['performance', 5], (n-24)/n)
        self.assertEqual(perf.loc['expected_count', 6], 24)
        self.assertEqual(perf.loc['real_count', 6], 24)
        self.assertAlmostEqual(perf.loc['performance', 6], 1.)
        self.assertEqual(perf.loc['expected_count', 7], n)
        self.assertEqual(perf.loc['real_count', 7], n-24)
        self.assertAlmostEqual(perf.loc['performance', 7], (n-24)/n)
        self.assertEqual(perf.loc['expected_count', 8], 24)
        self.assertEqual(perf.loc['real_count', 8], 24)
        self.assertAlmostEqual(perf.loc['performance', 8], 1.)

    def test_performance_inside_and_forward(self):
        n = 24*366
        perf = self.dataset.apply(TimeSeries.performance, limit_area=None, limit_direction='forward')
        self.assertEqual(self.dataset.shape[0], n)
        self.assertEqual(perf.loc['expected_count', 0], n)
        self.assertEqual(perf.loc['real_count', 0], n)
        self.assertAlmostEqual(perf.loc['performance', 0], 1.)
        self.assertEqual(perf.loc['expected_count', 1], n-24)
        self.assertEqual(perf.loc['real_count', 1], n-24)
        self.assertAlmostEqual(perf.loc['performance', 1], 1.)
        self.assertEqual(perf.loc['expected_count', 2], n)
        self.assertEqual(perf.loc['real_count', 2], 24)
        self.assertAlmostEqual(perf.loc['performance', 2], 24/n)
        self.assertEqual(perf.loc['expected_count', 3], 24)
        self.assertEqual(perf.loc['real_count', 3], 24)
        self.assertAlmostEqual(perf.loc['performance', 3], 1.)
        self.assertEqual(perf.loc['expected_count', 4], n)
        self.assertEqual(perf.loc['real_count', 4], n-24)
        self.assertAlmostEqual(perf.loc['performance', 4], (n-24)/n)
        self.assertEqual(perf.loc['expected_count', 5], n)
        self.assertEqual(perf.loc['real_count', 5], n-24)
        self.assertAlmostEqual(perf.loc['performance', 5], (n-24)/n)
        self.assertEqual(perf.loc['expected_count', 6], 6168)
        self.assertEqual(perf.loc['real_count', 6], 24)
        self.assertAlmostEqual(perf.loc['performance', 6], 24/6168)
        self.assertEqual(perf.loc['expected_count', 7], n)
        self.assertEqual(perf.loc['real_count', 7], n-24)
        self.assertAlmostEqual(perf.loc['performance', 7], (n-24)/n)
        self.assertEqual(perf.loc['expected_count', 8], 7368)
        self.assertEqual(perf.loc['real_count', 8], 24)
        self.assertAlmostEqual(perf.loc['performance', 8], 24/7368)

    def test_performance_dataframe_groupby(self):
        """Test performance can be applied with groupby"""
        r = self.dataset.groupby(TimeSeries.monthly).apply(TimeSeries.performance).swaplevel(axis=1)
        ec = r['expected_count']
        self.assertTrue(all(ec[0] == ec[[1, 2]].sum(axis=1)))
        self.assertTrue(all(ec[0] == ec[[3, 4]].sum(axis=1)))
        self.assertFalse(all(ec[0] == ec[[5, 6]].sum(axis=1)))  # Not at the end of the month
        self.assertTrue(all(ec[0] == ec[[7, 8]].sum(axis=1)))   # Exactly at the end of the month


class StatisticsTest(unittest.TestCase):
    """Test Time Statistics functions"""

    def setUp(self):
        """Create Log Normal Distribution for testing purposes"""
        self.t = pd.date_range('2020-01-01', '2021-01-01', freq='1H', closed='left')
        self.lognorm = stats.lognorm(s=1.15)
        self.xsample = self.lognorm.rvs(self.t.size)
        self.xrsample = np.round(self.xsample, decimals=2)
        self.xmsample = self.xsample.copy()
        self.xmsample[:self.t.size//10] = np.nan
        self.dataset = pd.DataFrame(np.array([self.xsample, self.xrsample, self.xmsample]).T, index=self.t)

    def test_ecdf_exception(self):
        """Test ECDF (numpy) on vectors"""
        with self.assertRaises(BadParameter):
            TimeSeries.ecdf(self.xmsample, mode='foo')
        with self.assertRaises(BadParameter):
            TimeSeries.ecdf(self.xmsample, rtype='foo')

    def test_ecdf_numpy(self):
        """Test ECDF (numpy) on vectors"""
        TimeSeries.ecdf(self.xsample, mode='ppf')
        TimeSeries.ecdf(self.xrsample, mode='ppf')
        TimeSeries.ecdf(self.xmsample, mode='ppf')
        with self.assertRaises(ValueError):
            TimeSeries.ecdf(self.xmsample, mode='ppf', dropna=False)

    def test_ecdf_rtype(self):
        """Test ECDF returned type"""
        self.assertIsInstance(TimeSeries.ecdf(self.xsample, rtype='serie'), pd.Series)
        self.assertIsInstance(TimeSeries.ecdf(self.xsample, rtype='dict'), dict)

    def test_ecdf_statsmodels(self):
        """Test ECDF (statmodels) on vectors"""
        TimeSeries.ecdf(self.xsample, mode='ecdf')
        TimeSeries.ecdf(self.xrsample, mode='ecdf')
        TimeSeries.ecdf(self.xmsample, mode='ecdf')
        TimeSeries.ecdf(self.xmsample, mode='ecdf', dropna=False)

    def test_ecdf_dataframe_apply(self):
        """Test ECDF function on dataframe with apply"""
        self.dataset.apply(TimeSeries.ecdf)
        self.dataset.apply(lambda x: TimeSeries.ecdf(x, mode='ppf'))
        self.dataset.apply(lambda x: TimeSeries.ecdf(x, mode='ecdf'))

    def test_ecdf_dataframe_groupby(self):
        """Test ECDF function on dataframe with apply"""
        r = self.dataset.groupby(TimeSeries.monthly).apply(TimeSeries.ecdf)


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
