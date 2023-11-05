from escnn.nn import FieldType
from escnn.gspaces import GSpace

from numbers import Number

import math

from typing import Tuple, Any

def calc_pool_output_shape(
        d: int,
        pool: Any,
        input_shape: Tuple[int, ...],
        out_type: FieldType,
) -> Tuple[int, ...]:

    assert len(input_shape) == 2 + d, (input_shape, d)

    if hasattr(pool, 'output_size'):
        output_size = get_nd_tuple(pool.output_size, d)
        return (input_shape[0], out_type.size, *output_size)

    b, c, *xyz_i = input_shape

    kernel_size = get_nd_tuple(pool.kernel_size, d)
    padding = get_nd_tuple(pool.padding, d)
    stride = get_nd_tuple(pool.stride, d)
    dilation = get_nd_tuple(getattr(pool, 'dilation', 1), d)

    # See online docs for `torch.nn.MaxPool2D`.
    xyz_o = tuple(
        (xyz_i[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1
        for i in range(d)
    )

    ceil_mode = getattr(pool, 'ceil_mode', False)
    round = math.ceil if ceil_mode else math.floor

    xyz_o = tuple(
        int(round(xyz_o[i]))
        for i in range(d)
    )

    return (b, out_type.size, *xyz_o)

def check_dimensions(in_type: FieldType, d: int) -> None:
    assert d in [2, 3], f"Only dimensionality 2 or 3 are currently suported by 'd={d}' found"

    assert isinstance(in_type.gspace, GSpace)
    assert in_type.gspace.dimensionality == d, (in_type.gspace.dimensionality, d)

def check_pointwise_ok(in_type: FieldType) -> None:
    for r in in_type.representations:
        assert 'pointwise' in r.supported_nonlinearities, \
            f"""Error! Representation "{r.name}" does not support pointwise non-linearities
            so it is not possible to pool each channel independently"""

def get_nd_tuple(scalar_or_tuple, d):
    if isinstance(scalar_or_tuple, Number):
        return (scalar_or_tuple,) * d
    else:
        return scalar_or_tuple

