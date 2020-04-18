import sys

from odapi.settings import settings


class GenericError(Exception):
    """Generic Package Exception must be subclassed not raised"""

    def __init__(self, message, **kwargs):
        super().__init__(message)
        self.__dict__.update(kwargs)
        settings.logger.error("[{}] {}: {}".format(type(self).__name__, message, kwargs))


class BadParameter(GenericError):
    """This exception stands for Bad Parameter error in method calls"""


class MissingFrequency(GenericError):
    """This exception stands when a frequency is needed but not found"""


def main():
    raise BadParameter("There is nothing about foo or bar", foo="bar")
    sys.exit(0)


if __name__ == "__main__":
    main()
