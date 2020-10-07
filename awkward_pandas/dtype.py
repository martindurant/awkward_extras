import pandas as pd
from pandas.api.extensions import ExtensionDtype
import awkward1 as ak


@pd.api.extensions.register_extension_dtype
class AwkardType(ExtensionDtype):
    name = 'awkward'
    type = ak.Array
    kind = 'O'
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise ValueError

    @classmethod
    def __from_arrow__(cls, arrow_array):
        from .series import AwkwardSeries
        ak_arr = ak.from_arrow(arrow_array)
        return AwkwardSeries.from_awkward(ak_arr)

    @property
    def _is_boolean(self) -> bool:
        return True

    @property
    def _is_numeric(self) -> bool:
        return True

    @classmethod
    def construct_array_type(cls):
        return AwkardType
