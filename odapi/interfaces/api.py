import sys
import abc
import pathlib
import json

import numpy as np
import pandas as pd

from odapi.settings import settings
from odapi.toolbox.generic import SettingsFile


class API(abc.ABC):
    """
    **Abstract Base Class:**
    This class stands for API Interface.
    This class provides generic bare logic to handle API data exchange.

    To create a new API from this interface:

      - Subclass the interface;
      - Populate required class members;
      - Implement abstract methods:

        - :py:meth:`odapi.interfaces.API._get_token`;
        - :py:meth:`odapi.interfaces.API.get_metadata`.

    There are two class members to populate:

    :param _setting_path: path to a JSON setting file or dictionary
    :type _setting_path: str, pathlib.Path, dict

    :param _primary_key: name of the Primary Key used to bind metadata and data on the underlying API
    :type _primary_key: str

    The snippet below shows how to proceed:

    .. code-block:: python

       import pandas as pd
       from odapi.interfaces import API

       class MyAPI(API):

            _primary_key = 'id'
            _settings_path = 'settings.json'

            def _get_token(self, **kwargs):
                # Call to underlying API to get authorization token
                return dict(...)

            def fetch(self, endpoint, token=None, params=None, mode='frame', **kwargs):
                # Generic method to get a resource from the underlying API
                # Use get_token method if a token is required
                return pd.DataFrame(...)

            def get_metadata(self, **kwargs):
                # Return metadata as a frame using fetch method
                return pd.DataFrame(...)
    """

    _settings_path = None
    _primary_key = 'channelid'

    def __init__(self, credentials=None):
        """
        Initialize API instance with credentials (optional)

        :param credentials: A dictionary of required fields to authenticate, defaults to None
        :type credentials: dict, str, pathlib.Path
        """
        self._credentials = SettingsFile.load(credentials)
        self._settings = SettingsFile.load(self._settings_path) or {}
        self._meta = None

    @property
    def credentials(self):
        """Return Stored Credentials"""
        return self._credentials

    @property
    def settings(self):
        """Return settings"""
        return self._settings

    @property
    def source(self):
        """Return source"""
        return self.settings['source']

    @property
    def model(self):
        """Return model"""
        return self.settings['model']

    @property
    def API(self):
        """Return source API"""
        return self.source["data"]["API"]

    @property
    def target(self):
        """Return source API target"""
        return self.API["target"].format(**self.credentials or {})

    @property
    def endpoint(self):
        """Return formated source API endpoints"""
        return {k: self.target + v for (k, v) in self.API["endpoint"].items()}

    @property
    def mapping(self):
        """Return Key Mapping"""
        return self.model['keys']['mapping']

    @property
    def translation(self):
        """Return Key Translation"""
        return self.model['keys']['translation']

    @property
    def tables(self):
        """Return Tables"""
        return self.model['tables']

    @abc.abstractmethod
    def _get_token(self, **kwargs):
        """
        **Abstract method:**
        This method must return a valid authorization token for subsequent underlying API calls.

        :param kwargs: Any parameters required by the authentication/authorization flow
        :type kwargs: unpacked dict

        :raises NotImplementedError: When no token flow is available.

        :return: Valid authorization token
        :rtype: str, dict, object
        """
        pass

    def get_token(self, credentials=None):
        """Get token from given or stored credentials"""
        if credentials is None:
            credentials = self.credentials
        return self._get_token(**credentials)

    @property
    def token(self):
        """Get token for stored credentials"""
        return self.get_token()

    @abc.abstractmethod
    def fetch(self, endpoint, token=None, params=None, mode='frame', **kwargs):
        """
        **Abstract method:**
        This method must return a resource from the underlying API.
        It could be a DataFrame, a JSON-like or a bytes object depending on the ``mode`` switch.

        :param endpoint: Valid endpoint key defined in ``source.data.API.endpoint``
        :type endpoint: str

        :param token: Valid token for underlying API as returned by :py:meth:`odapi.interfaces.TimeSerie.get_token`
        :type token: str, object

        :param params: Parameters to setup the request for resource to fetch
        :type params: dict

        :param mode: Type of returned resource, choices in ``{'frame', 'json', 'raw'}``, defaults to frame.
        :type mode: str

        :param kwargs: Any extra parameters required to control the fetch call
        :type kwargs: unpacked dict

        :return: A resource from the underlying API of type defined by ``mode`` switch
        :rtype: pandas.DataFrame, bytes or object
        """
        pass

    @abc.abstractmethod
    def get_metadata(self, **kwargs):
        """
        **Abstract method:**
        This method must return a frame of metadata from the underlying API.

        :param kwargs: Any parameters required to retrieve the metadata
        :type kwargs: unpacked dict

        :return: Metadata frame where rows are uniquely identified by a column named as defined in ``_primary_key``
        :rtype: pandas.DataFrame
        """
        pass

    @property
    def meta(self):
        """Shortcut for cached metadata"""
        if self._meta is None:
            self._meta = self.get_metadata()
        return self._meta

    def select(self, **filters):
        """Select from meta"""

        settings.logger.debug("Selection Keys: {}".format(filters))
        meta = self.meta

        def query(k, v, m=meta):
            if isinstance(v, str):
                return meta[k].str.match(v).values
            elif isinstance(v, list):
                return np.bitwise_or.reduce(np.array([query(k, v2, m=m) for v2 in v]), 0)

        queries = []
        for key in filters:
            queries.append(query(key, filters[key], m=meta))

        queries = np.bitwise_and.reduce(np.array(queries), 0)
        selection = meta.loc[queries, :]
        settings.logger.debug("Selected {} row(s): {}".format(selection.shape[0], set(selection[self._primary_key])))
        return selection


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
