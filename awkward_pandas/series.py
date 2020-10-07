import operator
import numpy as np
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

    def __array_ufunc__(self, *args, **kwargs):
        ak_arr = self.data.__array_ufunc__(self, *args, **kwargs)
        return self.from_awkward(ak_arr)

    @classmethod
    def __from_arrow__(cls, arr):
        ak_arr = ak.from_arrow(arr)
        return cls.from_awkward(ak_arr)

    def __arrow_array__(self):
        return ak.to_arrow(self.data)

    def setitem(self, indexer, value):
        raise ValueError

    @property
    def nbytes(self):
        return self.data.layout.nbytes

    def copy(self, deep=False):
        return self.from_awkward(self.data.copy())

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

    def __str__(self):
        return "Awkward Series: " + str(self.data)

    def __eq__(self, other):
        if isinstance(other, AwkwardSeries):
            return self.from_awkward(self.data == other.data)
        elif isinstance(other, ak.Array):
            return self.from_awkward(self.data == other)
        else:
            raise ValueError

    def equals(self, other):
        if isinstance(other, AwkwardSeries):
            other = other.data
        return ak.all(self.data == other)

    @property
    def _typ(self):
        return "dataframe"

    @property
    def columns(self):
        if self.layout.numfields >= 0:
            return self.layout.keys()
        else:
            return []

    @property
    def ndim(self):
        return 1

    def isna(self):
        return np.array(ak.operations.structure.is_none(self))

    def take(self, indices, *args, **kwargs):
        return self[indices]
