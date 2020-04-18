import sys
import unittest

import pandas as pd

from odapi.settings import settings


class TimeSerieAPITest:

    @classmethod
    def setUpClass(cls):
        """Create Meta and data Sample for testing purposes"""
        # Time Range:
        if not hasattr(cls, 'stop'):
            cls.stop = pd.Timestamp.utcnow().floor('1D') - pd.Timedelta('3D')
        if not hasattr(cls, 'start'):
            cls.start = cls.stop - pd.Timedelta('7D')
        # Meta, Records and Data:
        cls.meta = cls.get_sample_meta()
        cls.records = cls.get_sample_records()
        cls.data = cls.records.pivot_table(index='start', columns=cls.client._primary_key, values='value')
        # Statistics:
        cls.tmin = cls.records.start.min()
        cls.tmax = cls.records.start.max()

    @classmethod
    def get_sample_meta(cls):
        """Get sampled metadata"""
        key = cls.client._primary_key
        meta = cls.client.get_metadata()
        meta = meta.loc[meta[key].isin(cls.channel_sample)]
        settings.logger.debug("Sampled {} meta based on {}={}".format(meta.shape, key, cls.channel_sample))
        return meta

    @classmethod
    def get_sample_records(cls):
        """Get sampled data"""
        recs = cls.client.get_records(cls.channel_sample, start=cls.start, stop=cls.stop)
        return recs

    def test_sample_records_by_selection(self):
        """Test Records Selection is identical when sampling with meta selection"""
        recs = self.client.get_records(self.meta, start=self.start, stop=self.stop)
        self.assertTrue(self.records.equals(recs))

    def test_sample_records_with_timezone(self):
        """Test Records Selection when time range is specified with another timezone"""
        recs = self.client.get_records(self.channel_sample, start=self.start.tz_convert('CET'),
                                       stop=self.stop.tz_convert('CET'))
        self.assertTrue(self.records.equals(recs))

    def test_timerange_order(self):
        """Test Time Range of sampled records is properly closed to left [tmin,tmax)"""
        self.assertTrue(self.start <= self.tmin)
        self.assertTrue(self.tmin <= self.tmax)
        self.assertTrue(self.tmax <= (self.stop - self.freq))

    def test_timerange_extent(self):
        """Test records span to the correct time range extent"""
        dt = (self.tmax - self.tmin) + self.freq
        print(self.tmin, self.tmax)
        print(self.start, self.stop)
        self.assertEqual(dt, self.stop - self.start)

    def test_channels_extent(self):
        """Test all channel index have been mapped"""
        self.assertTrue(set(self.data.columns) == set(self.channel_sample))


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
