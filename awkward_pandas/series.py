from collections.abc import Iterable
import operator

import numpy as np
import awkward1 as ak
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin
from .dtype import AwkardType


class AwkwardSeries(ExtensionArray, ExtensionScalarOpsMixin):
    ndim = 1
    dtype = AwkardType()
    _accessors = ['ak']

    def __init__(self, ak_arr=None):
        if isinstance(ak_arr, type(self)):
            self.data = ak_arr.data  # no copy
        elif isinstance(ak_arr, ak.Array):
            self.data = ak_arr
        elif isinstance(ak_arr, str):
            self.data = ak.from_json(ak_arr)
        elif isinstance(ak_arr, Iterable):
            self.data = ak.from_iter(ak_arr)
        elif ak_arr is None:
            # empty series
            self.data = ak.Array(data=[])
        else:
            raise ValueError

    def __dask_tokenize__(self):
        # prevent dask hashing via asarray/tolist, which is very slow
        return str(id(self))  # require exact identity, since is immutable

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

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
        if len(args) == 1 and isinstance(args[0], (slice, Iterable)):
            return type(self)(self.data.__getitem__(*args))
        if isinstance(result, tuple):
            return self._box_scalar(result)
        elif getattr(result, "ndim", None) == 0:
            return self._box_scalar(result.item())
        else:
            return result

    def __array_ufunc__(self, *args, **kwargs):
        ak_arr = self.data.__array_ufunc__(self, *args, **kwargs)
        return type(self)(ak_arr)

    def __array__(self, dtype=None):
        # should ONLY be used for printing
        if isinstance(type, AwkardType):
            return self.data
        import numpy as np
        return np.array(self.data.tolist(), dtype="O")

    @classmethod
    def __from_arrow__(cls, arr):
        ak_arr = ak.from_arrow(arr)
        return cls(ak_arr)

    def __arrow_array__(self):
        return ak.to_arrow(self.data)

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
        return cls(ak.concatenate(to_concat))

    def tolist(self):
        return self.data.tolist()

    def argsort(self, **kwargs):
        return ak.argsort(self.data)

    def unique(self):
        raise ValueError

    def __repr__(self):
        return "AwkwardSeries( %s )" % str(self.data)

    def __str__(self):
        return str(self.data)

    def __eq__(self, other):
        if isinstance(other, AwkwardSeries):
            return type(self)(self.data == other.data)
        else:
            return self == type(self)(other)

    def equals(self, other):
        if isinstance(other, AwkwardSeries):
            other = other.data
        return ak.all(self.data == other)

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

    def astype(self, dtype, copy=False):
        # operates elementwise
        # if we really wanted a normal array out, would use ak.to_numpy
        if isinstance(dtype, AwkardType):
            if copy:
                return type(self)(self.data.copy())
            else:
                return self
        if dtype in [object, "O", "object"]:
            return self.tolist()
        return ak.values_astype(self.data, dtype)

    @property
    def ak(self):
        from .accessor import AwkwardAccessor
        return AwkwardAccessor(self)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        def op2(a, b):
            if hasattr(b, 'data'):
                b = b.data
            return cls(op(a.data, b))

        return op2


AwkwardSeries._add_arithmetic_ops()
AwkwardSeries._add_comparison_ops()
