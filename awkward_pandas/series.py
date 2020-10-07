import operator
import pandas as pd
import awkward1 as ak
from pandas.core.arrays import ExtensionArray
from .dtype import AwkardType


class AwkwardSeries(ExtensionArray):
    dtype = AwkardType

    @classmethod
    def from_awkward(cls, ak_arr):
        s = AwkwardSeries()
        s.data = ak_arr
        return s

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls.from_awkward(ak.from_iter(scalars))

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    @property
    def shape(self):
        return len(self.data),

    def __len__(self):
        return len(self.data)

    def __getitem__(self, *args):
        result = operator.getitem(self.data, *args)
        if isinstance(result, tuple):
            return self._box_scalar(result)
        elif result.ndim == 0:
            return self._box_scalar(result.item())
        else:
            return self.from_awkward(result)

    def setitem(self, indexer, value):
        raise ValueError

    @property
    def nbytes(self):
        return self.data.layout.nbytes

    def copy(self, deep=False):
        return type(self)(self.data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        to_concat = [tc.data if isinstance(tc, AwkwardSeries) else tc
                     for tc in to_concat]
        return cls.from_awkward(ak.concatenate(to_concat))

    def tolist(self):
        return self.data.tolist()

    def argsort(self, **kwargs):
        return self.data.argsort(**kwargs)

    def unique(self):
        raise ValueError
