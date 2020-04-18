import io
import sys
import zipfile

import pandas as pd
import requests

from odapi.interfaces.timeseries import API
from odapi.settings import settings


class WorldDataBank(API):

    def _get_token(self, **kwargs):
        raise NotImplementedError("No token required")

    def fetch(self, url, file=None, **kwargs):
        r = requests.get(url, stream=True)
        settings.logger.debug("{} [{}]: {}".format(r.request.method, r.status_code, r.url))
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = z.namelist()
        print(fname)
        if file:
            fname = [x for x in fname if x.startswith(file)]
        f = z.open(fname[0])
        return f

    def get_metadata(self, **kwargs):
        f = self.fetch("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv",
                        'API_SP.POP.TOTL_DS2_en_csv_v2')
        df = pd.read_csv(f, header=2)
        return df

    @property
    def population(self):
        return self.get_metadata()


def main():

    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 30

    c = WorldDataBank()
    print(c.meta)

    sys.exit(0)


if __name__ == "__main__":
    main()
