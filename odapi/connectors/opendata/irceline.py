import sys
import pathlib

import numpy as np
import pandas as pd
import requests

from odapi.interfaces.timeseries import TimeSeriesAPI
from odapi.settings import settings


class Irceline(TimeSeriesAPI):

    _settings_path = pathlib.Path(__file__).parent / 'resources/irceline.json'
    _primary_key = 'serieid'

    def __init__(self, credentials=None):
        """Initialize IRCELINE API"""
        super().__init__(credentials=credentials)

    def _get_token(self, **kwargs):
        raise NotImplemented("IRCELINE is an opendata API which does not require a token to connect")

    def fetch(self, endpoint, token=None, params=None, mode='frame', **kwargs):
        """Fetch method"""
        rep = requests.get(self.endpoint[endpoint].format(**params or {}))
        settings.logger.debug("{} [{}]: {}".format(rep.request.method, rep.status_code, rep.url))
        if mode == 'frame':
            return pd.json_normalize(rep.json())
        elif mode == 'json':
            return rep.json()
        else:
            return rep.content

    def get_metadata(self, **kwargs):
        """Get Metadata"""
        # Download Timeserie Metadata:
        df = self.fetch('metadata')
        # Clean columns:
        df = df.loc[:, self.mapping['timeseries'].keys()].rename(columns=self.mapping['timeseries'])
        # Casts & postprocessing
        df['measurekey'] = df['measurename'].replace(self.translation['measurekey'])
        df['serieunits'] = df['serieunits'].replace(self.translation['unitskey'])
        df['started'] = pd.to_datetime(df['start'], origin='unix', unit='ms', utc=True)
        df['stopped'] = pd.to_datetime(df['stop'], origin='unix', unit='ms', utc=True)
        df['sitename'] = df['sitekey'].apply(lambda x: "-".join(x.split('-')[1:]))
        df['sitekey'] = df['sitekey'].apply(lambda x: x.split('-')[0].strip())
        df['seriekey'] = df['measurekey'] + '/' + df['sitekey'] + ' (' + df['serieunits'] + ')'
        df['lat'] = df['geom'].apply(lambda x: x[0])
        df['lon'] = df['geom'].apply(lambda x: x[1])
        df = df.merge(self.table('sitetypes'), how='left')
        df = df.merge(self.table('factors'), how='left')
        settings.logger.debug("FRAME: {} metadata fetched".format(df.shape))
        return df.loc[:, ['serieid', 'siteid', 'measureid', 'serieunits', 'measurekey', 'measurename',
                          'sitekey', 'sitename', 'seriekey', 'molarmass', 'factor',
                          'siteloctype', 'sitesourcetype', 'lat', 'lon',
                          'nuts1id', 'nuts2id', 'nuts3id', 'nuts1name', 'nuts2name', 'nuts3name',
                          'lauid', 'launame',
                          'started', 'stopped']]

    def get_records(self, identifiers, start=None, stop=None, sentinel=-99.9):
        """Get Records"""
        # Irceline API select timestamp on end of sample period and include both sides (1 year at once max)
        params = self.prepare_parameters(identifiers, start=start, stop=stop, timezone='UTC')
        params = self.prepare_parameters(identifiers,
                                         start=params['start'] + pd.Timedelta('1H'),
                                         stop=params['stop'], freq='YS')
        dfs = []
        for sid in params['identifiers']:
            for (t0, t1) in zip(params['timerange'], params['timerange'][1:]):
                rep = self.fetch('records', params=dict(serieid=sid, start=t0, stop=t1), mode='json')
                try:
                    df = pd.DataFrame(rep['values'])
                    df['stop'] = pd.to_datetime(df['timestamp'], origin='unix', unit='ms')
                    df['start'] = df['stop'] - pd.Timedelta('1H')
                    df['serieid'] = sid
                    df['value'] = df['value'].astype(float)
                    dfs.append(df)
                except KeyError as err:
                    settings.logger.warning("Empty selection: serieid={} on [{};{}[".format(sid, t0, t1))
        if dfs:
            df = pd.concat(dfs).drop_duplicates().drop('timestamp', axis=1)
            # Bad missing data:
            if sentinel is not None:
                df.loc[df['value'] <= sentinel, 'value'] = np.nan
            # UTC Localize:
            df['start'] = df['start'].dt.tz_localize("UTC")
            df['stop'] = df['stop'].dt.tz_localize("UTC")
            settings.logger.debug("FRAME: {} record(s) fetched".format(df.shape[0]))
            return df[['serieid', 'start', 'stop', 'value']]

    @property
    def sites(self):
        df = self.meta.groupby(['siteid', 'sitekey', 'sitename', 'lon', 'lat']).agg({
            'serieid': list, 'measureid': set, 'measurekey': set,
            'started': 'min', 'stopped': 'max'
        })
        df['count'] = df['serieid'].apply(len)
        df = df.sort_values('sitekey').reset_index()
        return df.merge(self.table('sitetypes'), how='left')

    @property
    def measures(self):
        df = self.meta.groupby(['measureid', 'measurekey', 'measurename']).agg({
            'serieid': list, 'siteid': set, 'sitekey': set, 'serieunits': set,
            'started': 'min', 'stopped': 'max'
        })
        df['count'] = df['serieid'].apply(len)
        df = df.sort_values('measurekey').reset_index()
        return df.merge(self.table('factors'), how='left')

    @property
    def events(self):
        df = pd.DataFrame(self.tables['events'])
        df['start'] = pd.to_datetime(df['start'], utc=True)
        df['stop'] = pd.to_datetime(df['stop'], utc=True)
        df['start'] = df['start'].fillna(pd.Timestamp.min.tz_localize('UTC'))
        df['stop'] = df['stop'].fillna(pd.Timestamp.max.tz_localize('UTC'))
        return df


def main():
    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 30

    c = Irceline()
    s = c.select(measurekey=['NO', 'CO2'], sitekey='41.*')
    print(s)

    print(c.sites)

    sys.exit(0)


if __name__ == "__main__":
    main()
