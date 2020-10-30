import pandas as pd
from pandas.api.extensions import ExtensionDtype
import awkward1 as ak


class ADT_representor(type):

    def __repr__(self):
        return 'AwkwardDType'


@pd.api.extensions.register_extension_dtype
class AwkardType(ExtensionDtype, metaclass=ADT_representor):
    name = 'awkward'
    type = ak.Array
    kind = 'O'
    na_value = pd.NA

    @classmethod
    def construct_from_string(cls, string):
        return cls()

    @classmethod
    def __from_arrow__(cls, arrow_array):
        from .series import AwkwardSeries
        ak_arr = ak.from_arrow(arrow_array)
        return AwkwardSeries(ak_arr)

    @property
    def _is_boolean(self) -> bool:
        return True

    @property
    def _is_numeric(self) -> bool:
        return True

    @classmethod
    def construct_array_type(cls):
        from .series import AwkwardSeries
        return AwkwardSeries

    def __repr__(self):
        ## TODO: instance repr should be different from class one?
        return "AwkwardDType"
