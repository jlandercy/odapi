import sys
import abc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats

from odapi.settings import settings


class ODE:

    @staticmethod
    @abc.abstractmethod
    def ode(*args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def ivp(*args, **kwargs):
        pass

    @classmethod
    def fit(cls, t, y, **kwargs):
        popt, pcov = optimize.curve_fit(cls.ivp, t, y, **kwargs)
        return {'popt': popt, 'cov': pcov}

    @classmethod
    def regress(cls, t, y, **kwargs):
        popt, pcov = optimize.curve_fit(cls.ivp, t, y, **kwargs)
        yhat = cls.ivp(t, *popt)
        return {'popt': popt, 'pcov': pcov, 'yhat': yhat}

    @staticmethod
    def test(x, fexp, ddof=0):
        chi2 = stats.chisquare(x, f_exp=fexp, ddof=ddof)
        pdiv = stats.power_divergence(x, f_exp=fexp, lambda_="log-likelihood")
        return {'chi2': chi2, 'pdiv': pdiv}


class GGM(ODE):

    @staticmethod
    def ode(t, C, r, p):
        return r*np.power(C, p)

    @staticmethod
    def ivp(t, C, r, p):
        return integrate.solve_ivp(GGM.ode, (t[0], t[-1]), [C], t_eval=t, args=(r, p)).y[0]


class GRM(ODE):

    @staticmethod
    def ode(t, C, r, p, K, a):
        return r*np.power(C, p)*(1-np.power((C/K),a))

    @staticmethod
    def ivp(t, C, r, p, K, a):
        return integrate.solve_ivp(GRM.ode, (t[0], t[-1]), [C], t_eval=t, args=(r, p, K, a),
                                   method='DOP853', rtol=1e-6).y[0]


def main():
    sys.exit(0)


if __name__ == "__main__":
    main()
