import sys

import numpy as np
import pandas as pd

from odapi.settings import settings
from odapi.errors import BadParameter


class EnergyMeter:

    @staticmethod
    def distribute(df, freq='15T', mode='bfill'):
        """
        Equally distribute quantity over time axis:
        """
        modes = {'bfill': +1, 'ffill': -1}
        if mode not in modes:
            raise BadParameter("Mode must be in {}, received instead {}".format(modes.keys(), mode))
        df = df.sort_index()
        ts = pd.DataFrame(df.index.values, columns=['t1'])
        ts['t0'] = ts.t1.shift(modes[mode])
        ts['dt'] = (ts.t1 - ts.t0) * modes[mode]
        ts['c'] = ts.dt / pd.Timedelta(freq)
        df = df.div(ts.c.values, axis=0)
        df = df.resample(freq).agg(mode)
        return df


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
