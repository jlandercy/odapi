import sys
import unittest
from types import SimpleNamespace

from odapi.settings import settings


class SettingsTest(unittest.TestCase):

    def test_NameSpace(self):
        self.assertIsInstance(settings, SimpleNamespace)

    def test_RequiredSettings(self):
        self.assertTrue({'package', 'resources', 'uuid4'}.issubset(settings.__dict__))


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
