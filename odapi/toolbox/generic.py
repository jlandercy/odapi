import sys
import pathlib
import json
from functools import wraps

import numpy as np

from odapi.settings import settings


class SettingsFile:
    """
    Settings files
    """
    @staticmethod
    def load(path_or_object):
        """
        Load file or dict settings

        :param path_or_object: Valid path pointing to a valid JSON resource or dict
        :param path_or_object: str, pathlib.Path, dict

        :return: Dictionary of Settings
        :rtype: dict
        """
        if isinstance(path_or_object, str):
            path_or_object = pathlib.Path(path_or_object)
        if isinstance(path_or_object, dict) or path_or_object is None:
            path_or_object = path_or_object
        else:
            with path_or_object.open() as fh:
                settings.logger.debug("Load Settings {}".format(path_or_object))
                path_or_object = json.load(fh)
        return path_or_object


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
