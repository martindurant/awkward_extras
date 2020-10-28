import functools
import inspect
import pandas as pd
import awkward1 as ak
from .series import AwkwardSeries
from .dtype import AwkardType

funcs = [n for n in dir(ak) if inspect.isfunction(getattr(ak, n))]


@pd.api.extensions.register_series_accessor("ak")
class AwkwardAccessor:

    def __init__(self, pandas_obj):
        if not self._validate(pandas_obj):
            raise AttributeError("ak accessor called on incompatible data")
        self._obj = pandas_obj
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            if isinstance(self._obj, AwkwardSeries):
                self._arr = self._obj
            elif isinstance(self._obj.dtype, AwkardType) and isinstance(self._obj, pd.Series):
                # this is a pandas Series that contains an Awkward
                self._arr = self._obj.values
            elif isinstance(self._obj.dtype, AwkardType):
                # a dask series - figure out what to do here
                raise NotImplementedError
            else:
                # this recreates series, possibly by iteration
                self._arr = AwkwardSeries(self._obj)
        return self._arr

    @staticmethod
    def _validate(*_):
        return True

    def to_arrow(self):
        return self.arr.data.to_arrow()

    def cartesian(self, other, **kwargs):
        if isinstance(other, AwkwardSeries):
            other = other.data
        return AwkwardSeries(ak.cartesian([self.arr.data, other], **kwargs))

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
            ak_arr = func(self.arr.data, *others, **kwargs)
            # TODO: special case to carry over index and name information where output
            #  is similar to input, e.g., has same length
            if isinstance(ak_arr, ak.Array):
                # TODO: perhaps special case here if the output can be represented
                #  as a regular num/cupy array
                return AwkwardSeries(ak_arr)
            return ak_arr

        return f
