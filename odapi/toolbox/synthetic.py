import sys

import numpy as np
import pandas as pd

from odapi.settings import settings


class FakeSignal:

    @staticmethod
    def electrical():
        t1 = pd.Timestamp.utcnow().floor('1D')
        t0 = t1 - pd.Timedelta('1D')
        ts = pd.date_range(t0, t1, freq='10T', closed='left', name='time')
        tx = np.array(pd.to_timedelta(ts - t0).total_seconds() / (24 * 60 ** 2))
        A = np.array([1, 2, 3, 4, 5, 6]) * 2 * np.pi
        B = np.linspace(0, 1, A.size) * np.pi
        C = np.linspace(2, 5, A.size)
        tx = np.array([tx, ] * A.size)
        X = (np.sin(A.reshape(-1, 1) * tx + B.reshape(-1, 1)) * C.reshape(-1, 1)).T
        e = 0.1 * np.random.randn(*X.shape)
        cols = pd.MultiIndex.from_product([['Source-A', 'Source-B', 'Source-C'], ['I (A)', 'U (V)']],
                                          names=['meter', 'measure'])
        df = pd.DataFrame(X + e, index=ts, columns=cols)
        return df


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
