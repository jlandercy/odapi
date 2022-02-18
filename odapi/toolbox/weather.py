import sys
import collections

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from odapi.settings import settings
from odapi.errors import BadParameter


class Wind:
    """
    This class provides methods to handle Wind series.
    """

    @staticmethod
    def cardinals():
        """Return Major Cardinal Direction and related Goniometric angles"""
        return Wind.coordinates(order=1)[['label', 'coord']].set_index('label').to_dict()['coord']

    @staticmethod
    def coord_labels(order=3):
        """Return Wind Direction bin labels"""
        _labels = ["N", "NNE", "NE", "NEE", "E", "SEE", "SE", "SSE",
                   "S", "SSW", "SW", "SWW", "W", "NWW", "NW", "NNW"]
        return [s for s in _labels if len(s) <= order]

    @staticmethod
    def coord_bin(order=3):
        """
        Return coordinate bin measures

        Wind Rose decompose as :math:`n_\\mathrm{B} = 4 \\cdot 2^{r-1}` bins where :math:`r` is the order
        of coordinate detail (equals to the number of letters to code the coordinate label).

        Coordinate bin thus has a size of :math:`d_\\mathrm{B} = \\frac{360}{n_\\mathrm{B}}` goniometric degrees
        """
        nB = 4*(2**(order-1))
        dB = 360/nB
        return nB, dB

    @staticmethod
    def coord_bins(order=3, center=True):
        """Return coordinate bin limits"""
        n, db = Wind.coord_bin(order=order)
        b = np.linspace(0, 360, n+1)
        if center:
            return b - db/2
        else:
            return b

    @staticmethod
    def coordinates(order=3):
        """Create coordinate table"""
        df = pd.DataFrame({
            'label': Wind.coord_labels(order=order),
            'coord': Wind.coord_bins(order=order, center=False)[:-1],
            'lower': Wind.coord_bins(order=order, center=True)[:-1],
            'upper': Wind.coord_bins(order=order, center=True)[1:],
        })
        df['order'] = df['label'].apply(len)
        return df

    @staticmethod
    def coord_index(x, order=3):
        """Convert Goniometric angles (degrees) into Coordinate Index of order r (float to keep nan)"""
        if isinstance(x, pd.DataFrame):
            return x.apply(Wind.coord_index, order=order)
        elif isinstance(x, collections.abc.Iterable):
            return pd.Series(x, dtype=np.float).apply(Wind.coord_index, order=order)
        else:
            nB, dB = Wind.coord_bin(order=order)
            return np.floor(((x + dB/2) / dB) % nB)

    @staticmethod
    def direction(x, order=3):
        """Get Wind Direction expressed in Coordinate Indices or Labels"""
        tag = pd.Series(x)
        cidx = Wind.coord_index(tag.dropna(), order=order).astype(int)
        labels = np.array(Wind.coord_labels(order=order))
        tag.loc[~tag.isnull()] = labels[cidx]
        return tag

    @staticmethod
    def rad2deg(x):
        """Convert Radians to Degrees"""
        return np.rad2deg(x)

    @staticmethod
    def deg2rad(x):
        """Convert Degrees to Radians"""
        return np.deg2rad(x)

    @staticmethod
    def gonio2trigo_rad(x):
        """
        Convert Goniometric angles (rad) into trigonometric angles (rad)

        :math:`x_\\mathrm{trigo} = \\frac{\\pi}{2} - x_\\mathrm{gonio}`
        """
        return np.pi/2 - x

    @staticmethod
    def trigo2gonio_rad(x):
        """
        Convert Trigonometric angles (rad) into Goniometric angles (rad)

        :math:`x_\\mathrm{gonio} = \\frac{\\pi}{2} - x_\\mathrm{trigo}`
        """
        return np.pi/2 - x

    @staticmethod
    def trigo2gonio_deg(x):
        """
        Convert Trigonometric angles (degrees) into Goniometric angles (degrees)
        """
        return Wind.rad2deg(Wind.trigo2gonio_rad(Wind.deg2rad(x)))

    @staticmethod
    def gonio2trigo_deg(x):
        """
        Convert Goniometric angles (degrees) into Trigonometric angles (degrees)
        """
        return Wind.rad2deg(Wind.gonio2trigo_rad(Wind.deg2rad(x)))

    @staticmethod
    def radsat(x):
        """Saturate Radian angles to [0;2*pi)"""
        return x % (2*np.pi)

    @staticmethod
    def degsat(x):
        """Saturate Degrees angles to [0;360)"""
        return x % 360

    @staticmethod
    def colormap(cmap='Spectral_r', q=np.arange(0.0, 1.01, 0.1)):
        """
        Return a descrete colormap for Percentile Roses
        :param cmap: Color map theme
        :param q: Percentile bins boundaries
        :return: A colormap and a norm associated to percentiles
        """
        cmap = mpl.cm.get_cmap(cmap)
        cmaplist = [cmap(x) for x in q]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(q) - 1)
        norm = mpl.colors.BoundaryNorm(q, cmap.N)
        return cmap, norm

    @staticmethod
    def boxplot(data, x, theta='WD/41R001 (°G)'):
        """
        Return distribution by wind direction
        :param data:
        :param x:
        :param theta:
        :return:
        """
        wd = data[[theta]].apply(Wind.direction)
        bo = Wind.coordinates()
        fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [7, 3]})
        pair = pd.concat([data[x], wd], axis=1).dropna()
        pair.boxplot(x, by=theta, positions=bo.sort_values('label').index, rot=90, ax=axes[0])
        pair.groupby(theta)[x].count()[bo.label].plot(kind='bar', ax=axes[1])
        fig.suptitle('')
        axes[0].set_title("Distribution by Wind Directions")
        axes[0].set_ylabel(x)
        axes[1].set_ylabel("Count")
        axes[1].set_xlabel("Wind Direction")
        axes[1].grid()
        return axes

    @staticmethod
    def rose(data, x, theta='WD/41R001 (°G)', qbins=True, points=False, means=True, cbar=True,
             q=np.arange(0.0, 1.01, 0.1), cmap='Spectral_r', figsize=(8, 6),
             mode="polygon", edgecolor="white", linewidth=0.0):
        """
        Return polar axe with percentile rose, points and means

        :param data: DataFrame holding time series including Wind Directions
        :param x: DataFrame Key to plot on Percentile Rose
        :param theta: DataFrame Key to identify Wind Directions series in Goniometric Degrees
        :param qbins: Draw Percentile bins
        :param points: Draw experimental points on Rose
        :param means: Draw means on Rose
        :param cbar: Draw color bar beside rose
        :param q: Percentile bins boundaries
        :param cmap: Colormap theme
        :param figsize: Figure Size
        :param mode: Plot mode
        :param edgecolor: Bar edge color
        :param linewidth: Bar linewidth

        :return: An axe holding the Percentile Rose
        """

        # Color Map:
        cmap, norm = Wind.colormap(cmap=cmap, q=q)

        # Create Polar Axis (projection cannot be changed after axe creation):
        fig, axe = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        # Removed warning:
        # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
        ticks = axe.get_xticks().tolist()
        axe.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        axe.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'], fontsize=10)
        axe.set_title(x)

        # Add Color Bar:
        if cbar:
            axb = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            cbar = mpl.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm)
            cbar.set_label("Percentile Bins")

        # Bins info:
        b = Wind.gonio2trigo_rad(Wind.deg2rad(Wind.coord_bins()))
        db = b[1] - b[0]

        # Draw Points:
        if points:
            axe.plot(data[theta].apply(Wind.deg2rad).apply(Wind.gonio2trigo_rad),
                     data[x], '.', markersize=0.35, color='k')

        # Draw Means:
        if means:
            m = data.groupby(Wind.coord_index(data[theta]))[x].mean()
            axe.plot(b[:-1] + db / 2, m, color='green', marker='o', markersize=3, linewidth=0)

        # Draw Percentiles Rose:
        if qbins:

            # Compute Rose:
            w = data[theta].apply(Wind.coord_index).replace({-1: np.nan}).rename('bins')
            r = pd.concat([data[x], w], axis=1).dropna()
            g = r.groupby('bins').quantile(q).unstack()

            # Render Rose:
            for i in range(g.shape[0]):
                for j in range(g.shape[1] - 1):
                    col = cmap(1.1 * g.columns.levels[1][j])
                    if mode == "polygon":
                        # Polygon version (reference):
                        axe.fill([b[i], b[i], b[i + 1], b[i + 1]],
                                 [g.iloc[i, j], g.iloc[i, j + 1], g.iloc[i, j + 1], g.iloc[i, j]],
                                 color=col, edgecolor=edgecolor, linewidth=linewidth)
                    elif mode == "bar":
                        # Buggy (bar radius is not correct):
                        axe.bar(b[i]+db/2, g.iloc[i, j+1], width=db, bottom=g.iloc[i, j],
                                color=col, edgecolor=edgecolor, linewidth=linewidth)
                    else:
                        raise ValueError("Rose mode must be in {polygon, bar}")

        return axe


class Humidity:
    """
    This class provides methods yo handle Humidity series.
    """
    pass


class Sun:
    """
    This class provides methods to cope with Sun properties.
    """
    def __init__(self, city=None):
        """
        Initialize Sun class

        :param city: City name
        :type city: str
        """
        from astral.geocoder import database, lookup
        if city is None:
            city = 'Brussels'
        if isinstance(city, str):
            city = lookup(city, database())
        self._city = city

    @property
    def city(self):
        """Return city object"""
        return self._city

    def solar_day(self, t, mode='dict'):
        """
        This method returns day information for the given timestamp at the given location

        :param t: Timestamp when the information must be computed
        :type t: str, datetime.datetime, pandas.Timestamp

        :param mode: Return type (must be in {'astral', 'dict', 'serie'}), defaults to dict
        :type mode: str

        :return: Day information for the given timestamp
        :rtype: astral.sun.sun, dict, pandas.Series
        """
        from astral import sun
        modes = {'astral', 'dict', 'serie'}
        if mode not in modes:
            raise BadParameter("Mode must be in {}, received '{}' instead".format(modes, mode))
        s = sun.sun(self.city.observer, date=t)
        if mode == 'astral':
            return s
        elif mode == 'dict':
            return dict(s)
        else:
            return pd.Series(dict(s), index=[t])

    def solar_events(self, start, stop):
        """
        This method returns day information on a given time range as events table

        :param start:
        :param stop:

        :return: Events frame with solar information
        :rtype: pandas.DataFrame
        """
        ds = []
        ts = pd.date_range(start, end=stop, freq='1D').floor('1D')
        for t in ts:
            ds.append(self.solar_day(t, mode='dict'))
        return pd.DataFrame(ds, index=ts)


class Weather(Wind, Humidity, Sun):
    """
    This class provides methods to handle Wind, Humidity and Sun series.
    """
    pass


def main():
    m = Weather()

    print(Weather.cardinals())
    print(Weather.coordinates(order=1)[['label', 'coord']].to_dict())
    print(Weather.coordinates(order=2)[['label', 'coord']].to_dict())
    print(Weather.coordinates(order=3)[['label', 'coord']].to_dict())

    print(Weather.coord_index(np.nan))
    print(Weather.coord_index(100))
    print(Weather.coord_index([]))
    sys.exit(0)


if __name__ == "__main__":
    main()
