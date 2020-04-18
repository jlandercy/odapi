import sys
import abc
import json

from odapi.settings import settings
from odapi.interfaces.api import API


class GeomaticAPI(API):
    """
    **Abstract Base Class:**
    This class stands for Geomatic Interface, it extends API class.
    This class provides generic bare logic to handle Geomatic over API.

    To create a new Geomatic from this interface:

      - Subclass the interface;
      - Populate required class members (see class :py:class:`odapi.interfaces.API`);
      - Implement abstract methods: :py:meth:`odapi.interfaces.Geomatic.get_geometries`
                                    (also see class :py:class:`odapi.interfaces.API`).

    The snippet below shows how to proceed:

    .. code-block:: python

       import geopandas as gpd
       from odapi.interfaces import Geomatic

       class MyAPI(Geomatic):

            # ... See API Interface for complete implementation ...

            def get_geometries(self, identifiers, start, stop, **kwargs):
                # Return geometries as a frame using fetch method
                return gpd.GeoDataFrame(...)
    """

    @abc.abstractmethod
    def get_geometries(self, identifiers, **kwargs):
        """Get Geometries"""
        pass

    def geomatic(self, key, mode='json'):
        """Return GeoJSON as dict of frame"""
        assert mode in ('json', 'geojson', 'frame')
        path = settings.resources / 'geomatic/{}.geojson'.format(key)
        with path.open() as fh:
            data = json.load(fh)
        if mode == 'frame':
            import geopandas as gpd
            g = gpd.GeoDataFrame.from_features(data['features'])
        return data


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
