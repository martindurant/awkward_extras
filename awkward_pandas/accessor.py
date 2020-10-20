import functools
import inspect
import pandas as pd
import awkward1 as ak
from .series import AwkwardSeries

funcs = [n for n in dir(ak) if inspect.isfunction(getattr(ak, n))]


@pd.api.extensions.register_series_accessor("ak")
class AwkwardAccessor:

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(getattr(obj, "data", None), ak.Array)

    def to_arrow(self):
        return self._obj.data.to_arrow()

    def cartesian(self, other, **kwargs):
        if isinstance(other, AwkwardSeries):
            other = other.data
        return AwkwardSeries.from_awkward(ak.cartesian([self._obj.data, other], **kwargs))

    def __getattr__(self, item):
        from .series import AwkwardSeries
        # replace with concrete implementations of all top-level ak functions
        if item not in funcs:
            raise AttributeError
        func = getattr(ak, item)

        @functools.wraps(func)
        def f(*others, **kwargs):
            others = [other.data if isinstance(getattr(other, "data", None), ak.Array) else other
                      for other in others]
            ak_arr = func(self._obj.data, *others, **kwargs)
            if isinstance(ak_arr, ak.Array):
                return AwkwardSeries.from_awkward(ak_arr)
            return ak_arr

        return f
