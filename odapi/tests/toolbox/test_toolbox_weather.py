import sys
import collections
import unittest

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

from odapi.toolbox.weather import Wind, Humidity, Sun, Weather

pd.options.display.max_rows = 200
pd.options.display.max_columns = 10


class WindTest(unittest.TestCase):
    """Test Wind Direction Arithmetic"""

    def setUp(self):

        # Angles Arithmetic:
        self.trigo_degrees = np.linspace(-90, 450, 5401)
        self.trigo_radians = np.linspace(-np.pi/2, 5/2*np.pi, 5401)
        self.gonio_radians = np.pi/2 - self.trigo_radians
        self.gonio_degrees = self.gonio_radians*180/np.pi

        # Wind Directions:
        self.gonio_cycles = np.arange(-2 * 360, 2 * 360, 11.25)

    def test_angle_conversion_idempotence(self):
        """Test conversion functions composition are idempotent"""
        self.assertTrue(np.allclose(self.trigo_radians, Wind.deg2rad(Wind.rad2deg(self.trigo_radians))))
        self.assertTrue(np.allclose(self.trigo_degrees, Wind.rad2deg(Wind.deg2rad(self.trigo_degrees))))
        self.assertTrue(np.allclose(self.trigo_radians, Wind.gonio2trigo_rad(Wind.trigo2gonio_rad(self.trigo_radians))))
        self.assertTrue(np.allclose(self.trigo_degrees, Wind.gonio2trigo_deg(Wind.trigo2gonio_deg(self.trigo_degrees))))

    def test_angle_conversion(self):
        """Test conversion functions return expected results"""
        self.assertTrue(np.allclose(self.trigo_radians, Wind.deg2rad(self.trigo_degrees)))
        self.assertTrue(np.allclose(self.trigo_degrees, Wind.rad2deg(self.trigo_radians)))
        self.assertTrue(np.allclose(self.gonio_radians, Wind.trigo2gonio_rad(self.trigo_radians)))
        self.assertTrue(np.allclose(self.gonio_degrees, Wind.trigo2gonio_deg(self.trigo_degrees)))
        self.assertTrue(np.allclose(self.trigo_radians, Wind.gonio2trigo_rad(self.gonio_radians)))
        self.assertTrue(np.allclose(self.trigo_degrees, Wind.gonio2trigo_deg(self.gonio_degrees)))

    def test_cardinals(self):
        """Test Major Cardinal Points"""
        self.assertEqual(Wind.cardinals(), {'N': 0.0, 'E': 90.0, 'S': 180.0, 'W': 270.0})

    def test_coordinates(self):
        """Test Coordinate Table is correct"""
        c1 = pd.DataFrame({'label': {0: 'N', 1: 'E', 2: 'S', 3: 'W'},
                           'coord': {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}})
        c2 = pd.DataFrame({'label': {0: 'N', 1: 'NE', 2: 'E', 3: 'SE', 4: 'S', 5: 'SW', 6: 'W', 7: 'NW'},
                           'coord': {0: 0.0, 1: 45.0, 2: 90.0, 3: 135.0, 4: 180.0, 5: 225.0, 6: 270.0, 7: 315.0}})
        c3 = pd.DataFrame({'label': {0: 'N', 1: 'NNE', 2: 'NE', 3: 'NEE', 4: 'E', 5: 'SEE', 6: 'SE', 7: 'SSE',
                                     8: 'S', 9: 'SSW', 10: 'SW', 11: 'SWW', 12: 'W', 13: 'NWW', 14: 'NW', 15: 'NNW'},
                           'coord': {0: 0.0, 1: 22.5, 2: 45.0, 3: 67.5, 4: 90.0, 5: 112.5, 6: 135.0, 7: 157.5, 8: 180.0,
                                     9: 202.5, 10: 225.0, 11: 247.5, 12: 270.0, 13: 292.5, 14: 315.0, 15: 337.5}})
        self.assertTrue(Wind.coordinates(order=1)[['label', 'coord']].equals(c1))
        self.assertTrue(Wind.coordinates(order=2)[['label', 'coord']].equals(c2))
        self.assertTrue(Wind.coordinates(order=3)[['label', 'coord']].equals(c3))

    def test_coord_bins_constrained_injective_mapping(self):
        """Test if all values are mapped to all bins (injective function) and if function respects image domain"""
        for r in range(1, 4):
            nB, dB = Wind.coord_bin(order=r)
            idx = Wind.coord_index(self.gonio_cycles, order=r)
            self.assertTrue(all((idx >= 0) & (idx < nB)))
            self.assertEqual(set(range(nB)), set(idx.astype(int)))

    def test_coord_bins_mapping_correctness(self):
        """Test mapping provide correct answers when applied on well know goniometric angles"""
        coords = Wind.coordinates()
        coords['tag'] = coords['coord'].apply(Wind.direction)
        self.assertTrue(coords['tag'].equals(coords['label']))
        #print(coords)

    def test_coord_bins_mapping_representativeness(self):
        """Test if mapping is properly distributed (all coordinate have same representativeness)"""
        for r in range(1, 4):
            nB, dB = Wind.coord_bin(order=r)
            idx = Wind.coord_index(self.gonio_cycles, order=r)
            c = collections.Counter(idx)
            self.assertTrue(all([abs(c[i] - c[j]) < 1 for (i, j) in zip(range(nB), range(1, nB))]))

    def test_direction(self):
        """Test direction mapping for goniometric angle"""
        for r in range(1, 4):
            tags = Wind.direction(self.gonio_cycles, order=r)
            #print(pd.DataFrame({'gonio': self.gonio_cyles, 'tags': tags}))

    def test_scalar_direction(self):
        """Test direction mapping from goniometric angle when NaN"""
        tag = Wind.direction(180)
        # print(tag)

    def test_missing_direction(self):
        """Test direction mapping from goniometric angle when NaN"""
        cycles = self.gonio_cycles.copy()
        cycles[cycles.size//2:] = np.nan
        for r in range(1, 4):
            tags = Wind.direction(cycles, order=r)
            index = Wind.coord_index(cycles, order=r)
            #print(pd.DataFrame({'gonio': self.gonio_cycles, 'tags': tags, 'index': index}))


class WindPlots(unittest.TestCase):
    """Test Wind Direction Arithmetic"""

    def setUp(self):
        self.theta = np.arange(0, 180, 360/180)
        self.data = np.arange(self.theta.size)
        self.frame = pd.DataFrame({"WD": self.theta, "x": self.data})

    def tearDown(self) -> None:
        plt.show()

    def generate_frame(self, n=100, x_dist=stats.norm(), t_dist=stats.uniform(scale=360)):

        theta = t_dist.rvs(size=n)
        data = x_dist.rvs(size=n)

        frame = pd.DataFrame({
            "theta": theta,
            "data": theta*data
        })
        return frame

    def test_random_frame(self):
        x = self.generate_frame(n=500)
        axe = Wind.boxplot(x, 'data', theta="theta")
        axe = Wind.rose(x, 'data', theta="theta", frequencies=np.arange(0.1, 0.91, 0.1))
        axe = Wind.rose(x, 'data', theta="theta", quantiles=False, points=True)


    def test_boxplot(self):
        axe = Wind.boxplot(self.frame, 'x', theta='WD')
        #plt.show()

    def test_windrose(self):
        axe = Wind.rose(self.frame, 'x', theta='WD')


def main():
    unittest.main()
    sys.exit(0)


if __name__ == "__main__":
    main()
