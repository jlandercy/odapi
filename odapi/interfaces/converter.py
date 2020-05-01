import sys
import abc

from odapi.settings import settings


class Converter(abc.ABC):
    """Converter Base Class"""

    @staticmethod
    @abc.abstractmethod
    def to_frame(result, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def to_product(frame, **kwargs):
        pass


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
