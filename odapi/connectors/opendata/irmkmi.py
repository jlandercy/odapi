import sys
import pathlib

import numpy as np
import pandas as pd
import requests

from odapi.interfaces.timeseries import TimeSeriesAPI
from odapi.settings import settings


class IRMKMI(TimeSeriesAPI):

    _settings_path = pathlib.Path(__file__).parent / 'resources/irmkmi.json'
    _primary_key = 'serieid'

    def __init__(self, credentials=None):
        """Initialize IRCELINE API"""
        super().__init__(credentials=credentials)

    def _get_token(self, **kwargs):
        raise NotImplemented("IRMKMI is an opendata API which does not require a token to connect")

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
        settings.logger.debug("FRAME: {} metadata fetched".format(df.shape))
        return df

    def get_records(self, identifiers, start=None, stop=None, span=None, sentinel=-99.9):
        """Get Records"""
        # Irceline API select timestamp on end of sample period and include both sides (1 year at once max)
        params = self.prepare_parameters(identifiers, start=start, stop=stop, span=span, timezone='UTC')
        return

    @property
    def sites(self):
        df = self.meta.groupby(['siteid', 'sitekey', 'sitename', 'lon', 'lat']).agg({
            'serieid': list, 'measureid': set, 'measurekey': set,
            'started': 'min', 'stopped': 'max'
        })
        df['count'] = df['serieid'].apply(len)
        df = df.sort_values('sitekey').reset_index()
        df = df.merge(self.table('sitetypes'), how='left')
        return df

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


"""
https://opendata.meteo.be/service/ows?service=WFS&version=2.0.0&request=GetFeature&typenames=aws:aws_10min&outputformat=json&CQL_FILTER=((BBOX(the_geom,3.201846,50.193663,5.255236,51.347375,%20%27EPSG:4326%27))%20AND%20(timestamp%20%3E=%20%272020-08-01%2005:00:00%27%20AND%20timestamp%20%3C=%20%272020-08-14%2005:00:00%27))&sortby=timestamp
"""

def main():
    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 30

    c = IRMKMI()
    s = c.select(measurekey=['NO', 'CO2'], sitekey='41.*')
    print(s)

    print(c.sites)

    sys.exit(0)


if __name__ == "__main__":
    main()
