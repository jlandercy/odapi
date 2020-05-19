import sys
import abc

import dateutil
import numpy as np
import pandas as pd

# Statistics Toolboxes used in this module:
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from odapi.settings import settings
from odapi.errors import BadParameter, MissingFrequency


class TimeSeries:
    """
    Time Series Toolbox
    Collections of staticmethod useful when dealing with Time Series.
    """

    @staticmethod
    def coerce_timezone(t, timezone='UTC'):
        """
        This method coerces a Timestamp to ensure it is Time Zone aware

        :param t: Valid Timestamp (always prefer ISO-8601 standards)
        :type t: str, datetime.datetime, pandas.Timestamp

        :param timezone: Valid Time Zone Identifier, defaults to 'UTC'
        :type timezone: str

        :return: Timestamp Time Zone aware (converted or localized) for the given Time Zone
        :rtype: pandas.Timestamp
        """
        t = pd.Timestamp(t)
        try:
            return t.tz_convert(timezone)
        except TypeError:
            return t.tz_localize(timezone)

    @classmethod
    def prepare_parameters(cls, identifiers, start=None, stop=None, span=None, freq=None, floor='1T', key=None,
                           timezone='UTC'):
        """
        This method prepares parameters to setup selection and ease underlying API calls.

        :param identifiers: Selection of any valid primary keys (single or sequence of it) values available
                            in underlying API.  It also can be a metadata frame or a selection of rows from metadata.
        :type identifiers: int, str, sequences, pandas.DataFrame

        :param start: Timestamp for the lower bound of left-closed selection time range [start, stop)
        :type start: str, datetime.datetime, pandas.Timestamp

        :param stop: Timestamp for the upper bound of left-closed selection time range [start, stop)
        :type stop: str, datetime.datetime, pandas.Timestamp

        :param span: Timedelta representing the span of the selection time range [stop-span, span)
        :type span: str, datetime.timedelta, pandas.Timedelta

        :param freq: Time Frequency for interval happening with the time range [start, start+1*freq, ..., start+(n-1)*freq, stop)
        :type freq: str

        :param floor: Time Frequency to floor start timestamp (truncation), defaults to '1T'
        :type floor: str

        :param key: Alternative Primary Key name
        :type key: str

        :param timezone: Valid Time Zone Identifier, defaults to 'UTC'
        :type timezone: str

        :return: Prepared parameters: identifiers list or frame, start timestamp, stop timestamp and,
                                      time ranges equally split by time frequency
        :rtype: dict
        """

        # Default Parameters:
        if key is False:
            identifiers = None
        elif isinstance(identifiers, (int, str)):
            identifiers = set([identifiers])
        elif isinstance(identifiers, pd.DataFrame):
            identifiers = set(identifiers[key or cls._primary_key])

        if span is None:
            span = pd.Timedelta("14 days")
        elif isinstance(span, str):
            span = pd.Timedelta(span)

        if stop is None:
            stop = pd.Timestamp.utcnow().floor(floor)
        if start is None:
            start = stop - span

        start = TimeSeries.coerce_timezone(start, timezone)
        stop = TimeSeries.coerce_timezone(stop, timezone)

        trange = pd.date_range(start, end=stop, freq=freq, normalize=True, closed='left')
        trange = pd.DatetimeIndex([start] + list(trange) + [stop]).unique()

        res = {
            'identifiers': identifiers,
            'stop': stop,
            'start': start,
            'timerange': trange
        }

        settings.logger.debug("PARAM: {}".format(res))
        return res

    @abc.abstractmethod
    def get_records(self, identifiers, start=None, stop=None, **kwargs):
        """Get Records"""
        pass

    @property
    def events(self):
        """Placeholder for Event Table"""
        raise NotImplementedError("Missing events table, you must provide either a table or an implementation")

    @staticmethod
    def daily(x):
        """
        Map timestamp onto beginning of the day.

        :param x: A valid timestamp object.
        :type x: datetime.datetime, pandas.Timestamp

        :return: A timestamp mapped to the first day of the day.
        :rtype: datetime.datetime, pandas.Timestamp
        """
        return x.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def weekly(x):
        """
        Map timestamp onto a generic week that keeps day of week unchanged (useful for daily profile).
        Monday 1940-01-01 to Sunday 1940-01-07 is a good candidate for this mapping.

        :param x: A valid timestamp object.
        :type x: datetime.datetime, pandas.Timestamp

        :return: A timestamp mapped to a day of generic week 1940-01-01/07.
        :rtype: datetime.datetime, pandas.Timestamp
        """
        return x.replace(year=1940, month=1, day=1+x.weekday())

    @staticmethod
    def monthly(x):
        """
        Map timestamp onto beginning of the month.

        :param x: A valid timestamp object.
        :type x: datetime.datetime, pandas.Timestamp

        :return: A timestamp mapped to the first day of the month.
        :rtype: datetime.datetime, pandas.Timestamp
        """
        return TimeSeries.daily(x).replace(day=1)

    @staticmethod
    def yearly(x):
        """
        Map timestamp onto beginning of the year.

        :param x: A valid timestamp object.
        :type x: datetime.datetime, pandas.Timestamp

        :return: A timestamp mapped to the first day of the year.
        :rtype: datetime.datetime, pandas.Timestamp
        """
        return TimeSeries.monthly(x).replace(month=1)

    @staticmethod
    def ecdf(x, bins=100, p=None, mode='ecdf', rtype='serie', dropna=True):
        """
        Estimate ECDF of Time Series (groupby allowed)

        :param x: Series or DataFrame
        :type x: list, numpy.array, pandas.Series, pandas.DataFrame

        :param bins: Number of bins to create the histogram (bins > 0), defaults to 100
        :type bins: int

        :param p: Vector of percentiles to estimate (0 <= p <= 1), defaults to None
        :type p: list, numpy.array, pandas.Series

        :param mode: Mode of estimation (ppf or ecdf), defaults to ecdf:

          - **ppf**: Estimate ECDF from the PPF of the histogram (scipy based);
          - **ecdf**: Estimate ECDF from the inverse monotone function using linear spline (statsmodels based).
        :type mode: str

        :param rtype: Control how ECDF is returned (serie or dict), defaults to serie:

          - **serie**: Return a Pandas serie with ordered interpolated values and required percentiles as index;
          - **dict**: Return a dict with all intermediates values (distribution, direct and inverse functions,
                      required percentiles and interpolated values).
        :type rtype: str

        :param dropna: Drop NaN values from data (ecdf mode is NaN tolerant), defaults True
        :type dropna: bool

        :return: ECDF estimation for the given series
        :rtype: pandas.Series, dict or pandas.DataFrame
        """
        # Parameter check/init:
        modes = {'ppf', 'ecdf'}
        if mode not in modes:
            raise BadParameter("ECDF mode must be in {}, received '{}' instead".format(modes, mode))
        rtypes = {'serie', 'dict'}
        if rtype not in rtypes:
            raise BadParameter("ECDF rtype must be in {}, received '{}' instead".format(rtypes, rtype))
        if p is None:
            p = np.linspace(0., 1., 1001)
        if isinstance(x, pd.DataFrame):
            # Apply on each serie (allow groupby)
            return x.apply(TimeSeries.ecdf)
        else:
            # Apply on serie:
            x = pd.Series(x)
            if dropna:
                x = x.dropna()
            p = np.array(p)
            # Computations:
            if mode == 'ppf':
                hist = np.histogram(x, bins=bins)
                dist = stats.rv_histogram(hist)
                fdir = dist.cdf
                finv = dist.ppf
                xinv = finv(p)
            elif mode == 'ecdf':
                try:
                    fdir = dist = ECDF(x)
                    finv = monotone_fn_inverter(fdir, dist.x)
                    xinv = finv(p)
                except ZeroDivisionError:
                    # Catch error when empty serie (allow groupby)
                    finv = fdir = dist = None
                    xinv = np.full(p.shape, np.nan)
            # Create Serie:
            s = pd.Series(xinv, index=p)
            if rtype == 'serie':
                return s
            else:
                return {'x': x, 'p': p, 'xinv': xinv, 'fdir': fdir, 'finv': finv, 'dist': dist, 'serie': s}

    @staticmethod
    def performance(x, limit_area='inside', limit_direction='forward', scenario=None):
        """
        Assess acquisition performance for Time Series (groupby allowed)

        :param x: Series or DataFrame with defined frequency (mandatory)
        :type x: list, numpy.array, pandas.Series, pandas.DataFrame

        :param scenario: Type of scenario to evaluate performance (must be in :py:data:`{None, 'strict', 'forward', 'backward'}`),
                         defaults to :py:data:`None` (allow custom performance definition).
                         When set, this setting overwrite :py:data:`limit_area` and :py:data:`limit_direction`.
                         Scenarii are defined as follow:

         - **strict**: Performance is evaluated within the interval from the first to the last observed records
                        (:py:data:`limit_area='inside'`)
         - **forward**: Performance is evaluated within the interval of first observed record to the end of frame
                        (:py:data:`limit_area=None` and :py:data:`limit_direction='forward'`)
         - **backward**: Performance is evaluated within the interval of beginning of the frame to the last observed record
                        (:py:data:`limit_area=None` and :py:data:`limit_direction='backward'`)
        :type scenario: str

        :param limit_area: Limit Area switch as defined in pandas.Series.interpolate
        :type limit_area: str

        :param limit_direction: Limit Direction switch as defined in pandas.Series.interpolate
        :type limit_direction: str

        :return: Acquisition performance statistics (w.r.t acquisition scenario)
        :rtype: pandas.Series, dict or pandas.DataFrame
        """
        scenarii = {None, 'strict', 'forward', 'backward'}
        if scenario not in scenarii:
            raise BadParameter("Scenario must be in {}, received '{}' instead".format(scenarii, scenario))
        if x.index.freq is None:
            raise MissingFrequency("DatetimeIndex must have a frequency set: resample timeseries first")
        # Create Scenario if defined:
        if scenario is not None:
            if scenario == 'strict':
                limit_area = 'inside'
            else:
                limit_area = None
                limit_direction = scenario
        if isinstance(x, pd.DataFrame):
            # Apply on each serie (allow groupby):
            return x.apply(TimeSeries.performance).unstack()
        else:
            # Apply on serie:
            xr = pd.Series(x).dropna()
            xf = x.interpolate(limit_area=limit_area, limit_direction=limit_direction).dropna()
            ne = xf.index.size
            nr = xr.count()
            nt = x.size
            if ne == 0:
                nr = ne = np.nan
            return pd.Series({
                'left': x.index[0],
                'right': x.index[-1],
                'start': xf.index.min(),
                'stop': xf.index.max(),
                'expected_count': ne,
                'real_count': nr,
                'total_count': nt,
                'performance': nr/ne,
                'fillfactor': nr/nt,
                'leftclosed': xr.index.min() == x.index[0],
                'rightclosed': xr.index.max() == x.index[-1]
            })

    @staticmethod
    def holidays(year, timezone='CET'):
        """This methods generates yearly Belgium Holidays events"""
        easter = dateutil.easter.easter(year)
        df = pd.DataFrame([
            # Legal Belgium Holidays
            {'label': 'New Year %d' % year, 'start': pd.Timestamp(year, 1, 1), 'tags': ['legal', 'holidays']},
            {'label': 'Easter Sunday %d' % year, 'start': easter, 'tags': ['holidays', 'christian']},
            {'label': 'Easter Monday %d' % year, 'start': easter + pd.Timedelta('1D'), 'tags': ['legal', 'holidays', 'christian']},
            {'label': 'Labor Day %d' % year, 'start': pd.Timestamp(year, 5, 1), 'tags': ['legal', 'holidays']},
            {'label': 'Ascension %d' % year, 'start': easter + pd.Timedelta('39D'), 'tags': ['legal', 'holidays', 'christian']},
            {'label': 'Pentecost Sunday %d' % year, 'start': easter + pd.Timedelta('49D'), 'tags': ['holidays', 'christian']},
            {'label': 'Pentecost Monday %d' % year, 'start': easter + pd.Timedelta('50D'), 'tags': ['legal', 'holidays', 'christian']},
            {'label': 'National Day %d' % year, 'start': pd.Timestamp(year, 7, 21), 'tags': ['legal', 'holidays']},
            {'label': 'Assumption Day %d' % year, 'start': pd.Timestamp(year, 8, 15), 'tags': ['legal', 'holidays', 'christian']},
            {'label': 'All Saints Day %d' % year, 'start': pd.Timestamp(year, 11, 1), 'tags': ['legal', 'holidays', 'christian']},
            {'label': '1918 Armistice %d' % year, 'start': pd.Timestamp(year, 11, 11), 'tags': ['legal', 'holidays']},
            {'label': 'Christmas %d' % year, 'start': pd.Timestamp(year, 12, 25), 'tags': ['legal', 'holidays', 'christian']},
            # School Holidays:
            {'label': 'Summer Holidays %d' % year, 'start': pd.Timestamp(year, 7, 1), 'start': pd.Timestamp(year, 9, 1), 'tags': 'no-school'},
        ])
        df['start'] = df['start'].dt.tz_localize(timezone)
        df['stop'] = df['start'] + pd.Timedelta('1D')
        return df

    @staticmethod
    def groupby_events(df, events, boolean=False):
        """This method creates a grouper for events w.r.t. DatetimeIndex of the DataFrame"""
        events = events.explode('tags')
        # Init Index:
        index = {}
        q0 = (df.index < df.index[0])
        # Iter tag grouped events table:
        eg = events.groupby('tags')[['start', 'stop']].agg(list)
        for row in eg.itertuples():
            q = q0.copy()
            for (t0, t1) in zip(row.start, row.stop):
                q |= (df.index >= t0) & (df.index < t1)
            index[row.Index] = q
        # Add weekday/weekend:
        index['weekend'] = df.index.weekday >= 5
        index['weekday'] = ~index['weekend']
        # Convert boolean vector to timeindex:
        if not boolean:
            for k in index:
                index[k] = df.index[index[k]]
        return index

    @staticmethod
    def tag_events(df, events):
        """This method tags DataFrame w.r.t to DatetimeIndex based on a events frame"""
        grouper = TimeSeries.groupby_events(df, events, boolean=True)
        keys = np.array(list(grouper.keys()))
        coded = np.array([grouper[k] for k in keys]).T
        tags = [list(keys[np.where(c)[0]]) for c in coded]
        return tags

    @staticmethod
    def reglin(x):
        """This method computes the Linear Regression (OLS) on each serie of a DataFrame"""
        if isinstance(x, pd.DataFrame):
            return x.apply(TimeSeries.reglin).unstack()
        else:
            x = x.dropna()
            # Numeric time in second and start at the beginning:
            t = x.index.values.astype('datetime64[ms]').astype(float)/1e3
            t = t - t[0]
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, x)
            except ValueError:
                slope, intercept, r_value, p_value, std_err = [np.nan]*5
            return pd.Series({
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            })

    @staticmethod
    def autocorr(x, fft=True, nlags=14 * 30, missing='conservative'):
        """This method estimates Auto-Correlation Function for each serie of a DataFrame"""
        if x.index.freq is None:
            raise MissingFrequency("DatetimeIndex must have a frequency set: resample timeseries first")
        return acf(x, fft=fft, nlags=nlags, missing=missing)

    @staticmethod
    def seasonal(x, **kwargs):
        """
        This methods performs Seasonal Decomposition on each serie of DataFrame.

        :math:`y(t) = T(t) + S(t) + e(t)`
        """
        if x.index.freq is None:
            raise MissingFrequency("DatetimeIndex must have a frequency set: resample timeseries first")
        x = x.interpolate(**kwargs).dropna()
        r = seasonal_decompose(x, **kwargs)
        return pd.DataFrame({
            'trend': r.trend,
            'seasonal': r.seasonal,
            'residual': r.resid
        }).unstack()


def main():
    x = pd.concat([TimeSeries.holidays(i) for i in range(2012, 2021)])
    print(x)
    sys.exit(0)


if __name__ == "__main__":
    main()
