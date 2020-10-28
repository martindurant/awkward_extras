from dask.dataframe.extensions import make_array_nonempty, make_scalar, register_series_accessor
from .dtype import AwkardType
from .series import AwkwardSeries
from .accessor import AwkwardAccessor, ak


@make_array_nonempty.register(AwkardType)
def _(*_):
    return AwkwardSeries([1])


@make_scalar.register(AwkardType)
def _(*_):
    return ak.from_iter([1])


register_series_accessor("ak")(AwkwardAccessor)
