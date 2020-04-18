import sys
import abc

import numpy as np
import pandas as pd

from odapi.settings import settings
from odapi.interfaces.api import API
from odapi.toolbox.timeseries import TimeSeries


class TimeSeriesAPI(API, TimeSeries):
    """
    **Abstract Base Class:**
    This class stands for Time Serie Interface, it extends API class.
    This class provides generic bare logic to handle Time Series over API.

    To create a new Time Serie from this interface:

      - Subclass the interface;
      - Populate required class members (see class :py:class:`odapi.interfaces.API`);
      - Implement abstract methods: :py:meth:`odapi.interfaces.TimeSerie.get_records`
                                    (also see class :py:class:`odapi.interfaces.API`).

    The snippet below shows how to proceed:

    .. code-block:: python

       import pandas as pd
       from odapi.interfaces import TimeSerie

       class MyAPI(TimeSerie):

            # ... See API Interface for complete implementation ...

            def get_records(self, identifiers, start, stop, **kwargs):
                # Return records as a frame using fetch method
                return pd.DataFrame(...)
    """

    @abc.abstractmethod
    def get_records(self, identifiers, start=None, stop=None, **kwargs):
        """Get Records"""
        pass

    @property
    def events(self):
        """Placeholder for Event Table"""
        raise NotImplementedError("Missing events table, you must provide either a table or an implementation")

    @property
    def limits(self):
        """Return limits from tables"""
        df = pd.DataFrame(self.tables['limits'])
        df['scale'] = df['scale'].apply(lambda x: np.array(x))
        df['norms'] = df['norms'].apply(lambda x: np.array(x))
        return df

    def scales(self, index='measurekey'):
        """Return scale from limits"""
        return self.limits.set_index(index).to_dict()['scale']


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
