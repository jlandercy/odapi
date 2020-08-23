import sys
import pathlib
import json

import numpy as np
import pandas as pd
import requests

from odapi.interfaces.timeseries import TimeSeriesAPI
from odapi.settings import settings


class IRMKMI(TimeSeriesAPI):

    _settings_path = pathlib.Path(__file__).parent / 'resources/irmkmi.json'
    _primary_key = 'siteid'

    def __init__(self, credentials=None):
        """Initialize IRCELINE API"""
        super().__init__(credentials=credentials)

    def _get_token(self, **kwargs):
        raise NotImplemented("IRMKMI is an opendata API which does not require a token to connect")

    def fetch(self, endpoint, token=None, params=None, mode='frame', **kwargs):
        """Fetch method"""
        rep = requests.get(self.endpoint[endpoint].format(**params or {}))
        settings.logger.debug("{} [{}]: {}".format(rep.request.method, rep.status_code, rep.url))
        # Normalize bytes:
        data = rep.json()
        for rec in data["features"]:
            rec["properties"]["qc_flags"] = json.loads(rec["properties"]["qc_flags"])

        if mode == 'frame':
            return pd.json_normalize(data["features"])
        elif mode == 'json':
            return data
        else:
            return rep.content

    def normalize(self, df):
        df = df.filter(regex='geometry\.|properties\.')
        df = df.melt(id_vars=["properties.code", "properties.timestamp", "geometry.coordinates"])
        df["measurekey"] = df["variable"].apply(lambda x: x.split('.')[-1])
        coords = df.pop("geometry.coordinates")
        df["site_lon"] = coords.apply(lambda x: x[0])
        df["site_lat"] = coords.apply(lambda x: x[1])
        df = df.rename(columns={"properties.code": "siteid",  "properties.timestamp": "timestamp"})
        df = df[df["measurekey"] != "bbox"]
        df = df[df["measurekey"] != "type"]
        df.pop("variable")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_metadata(self, **kwargs):
        """Get Metadata"""
        # Download Timeserie Metadata:
        df = self.fetch('metadata')
        df = self.normalize(df)
        df = df.groupby(["siteid", "measurekey"]).agg({"site_lon": "first", "site_lat": "first"})
        df = df.reset_index()
        settings.logger.debug("FRAME: {} metadata fetched".format(df.shape))
        return df

    def get_records(self, identifiers, start=None, stop=None, span=None):
        """Get Records"""
        # Irceline API select timestamp on end of sample period and include both sides (1 year at once max)
        params = self.prepare_parameters(identifiers, start=start, stop=stop, span=span, timezone='UTC')
        df = self.fetch('records', params=params)
        df = self.normalize(df).drop(["site_lon", "site_lat"], axis=1)
        return df


def main():
    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 30

    c = IRMKMI()
    m = c.get_metadata()
    print(m)

    r = c.get_records(None)
    print(r)

    sys.exit(0)


if __name__ == "__main__":
    main()
