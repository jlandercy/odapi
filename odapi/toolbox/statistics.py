import sys
import itertools

from scipy import stats
import pandas as pd

from odapi.settings import settings


class StatisticalTest:

    _available_test = {
        "student": "welch",
        "T-Test": "welch",
        "welch": (stats.ttest_ind, {"equal_var": False, "nan_policy": "omit"}),
        "kolmogorov": stats.ks_2samp,
        "kolmogorov-smirnov": "kolmogorov",
        "KS-Test": "kolmogorov"
    }

    @staticmethod
    def sample_keys(ref, exp, mode='product'):
        """
        Generate combination of reference and experimental keys
        :param ref:
        :param exp:
        :param mode:
        :return:
        """
        if mode == 'pairwise':
            keys = set(ref).intersection(set(exp))
            return tuple(p for p in zip(keys, keys))
        elif mode == 'product':
            return tuple(p for p in itertools.product(ref, exp))
        elif mode == 'combination':
            raise NotImplemented("Combination mode not implemented yet")
        else:
            raise KeyError("Unknown mode '{}'".format(mode))

    @staticmethod
    def get_test(test):
        if callable(test):
            return test, {}
        elif isinstance(test, str):
            if test in vars(stats):
                f = getattr(stats, test)
            elif test in StatisticalTest._available_test:
                f = StatisticalTest._available_test[test]
                if isinstance(f, str):
                    f = StatisticalTest._available_test[f]
            else:
                raise KeyError("Unknown test '{}'".format(test))
            if callable(f):
                return f, {}
            else:
                return f
        else:
            TypeError("Test must be either a function or a function name (str), received {} instead".format(type(test)))

    @staticmethod
    def apply(ref, exp, test='student', mode='product', extra=True, **params):
        """
        Apply Statistical Test on reference and experimental DataFrame
        :param ref:
        :param exp:
        :param test:
        :param mode:
        :param extra:
        :param params:
        :return:
        """
        func, fparams = StatisticalTest.get_test(test)
        params.update(fparams)
        results = []
        for kref, kexp in StatisticalTest.sample_keys(ref.columns, exp.columns, mode=mode):
            result = {"ref_key": kref, "exp_key": kexp, "test": test}
            if extra:
                result.update({
                    "ref_count": ref[kref].count(), "exp_count": exp[kexp].count(),
                    "ref_mean": ref[kref].mean(), "exp_mean": exp[kexp].mean(),
                    "ref_std": ref[kref].mean(), "exp_std": exp[kexp].std(),
                })
            tobj = func(ref[kref], exp[kexp], **params)
            result["class"] = tobj.__class__.__name__
            result["params"] = params
            if not isinstance(tobj, dict):
                tobj = {k: getattr(tobj, k) for k in tobj._fields}
            result.update(tobj)
            results.append(result)
        return pd.DataFrame(results)


def main():

    pd.options.display.max_columns = 30

    import numpy as np
    df = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    print(df)

    r1 = StatisticalTest.apply(df, df, mode="product", test="T-Test")
    print(r1)

    r2 = StatisticalTest.apply(df, df, mode="product", test="KS-Test")
    print(r2)

    r2 = StatisticalTest.apply(df, df, mode="product", test=stats.ttest_ind)
    print(r2)

    sys.exit(0)


if __name__ == "__main__":
    main()
