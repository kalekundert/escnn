
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule
from .utils import calc_pool_output_shape, check_dimensions, get_nd_tuple

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any, Union

from math import prod


__all__ = ["NormMaxPool"]

_MAX_POOLS = {
        2: F.max_pool2d,
        3: F.max_pool3d,
}

class _NormMaxPoolND(EquivariantModule):
    
    def __init__(self,
                 *,
                 d: int,
                 in_type: FieldType,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 ceil_mode: bool = False
                 ):
        r"""
        
        Max-pooling based on the fields' norms. In a given window of shape :attr:`kernel_size`, for each
        group of channels belonging to the same field, the field with the highest norm (as the length of the vector)
        is preserved.
        
        Except :attr:`in_type`, the other parameters correspond to the ones of :class:`torch.nn.MaxPool2d`.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape
            
        """

        super().__init__()

        check_dimensions(in_type, d)

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.kernel_size = get_nd_tuple(kernel_size, d)
        self.stride = get_nd_tuple(stride if stride is not None else kernel_size, d)
        self.padding = get_nd_tuple(padding, d)
        self.dilation = get_nd_tuple(dilation, d)
        self.ceil_mode = ceil_mode
        
        # For each representation size, an "index" which will select all 
        # channels belonging to said representations.  Each "index" could 
        # either by a list of (integer) index numbers, or a slice.
        self._indices = {}
        
        # For each representation size, whether all the representations of that 
        # size are located immediately adjacent to each other.
        _contiguous = {}

        last_stop = 0
        last_size = None

        for r in self.in_type.representations:
            if r.size != last_size:
                _contiguous[r.size] = r.size not in _contiguous
        
            start, stop = last_stop, last_stop + r.size
            self._indices.setdefault(r.size, []).extend(range(start, stop))

            last_size = r.size
            last_stop = stop

        # Use slices for contiguous fields.  Presumably this is a bit more 
        # efficient?
        for k, indices in self._indices.items():
            if _contiguous[k]:
                self._indices[k] = slice(min(indices), max(indices) + 1)
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Run the norm-based max-pooling on the input tensor
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type
        
        b, c, *xyzi = input.tensor.shape
        
        assert len(xyzi) == self.d
        
        b, c, *xyzo = self.evaluate_output_shape(input.tensor.shape)

        # compute the squares of the values of each channel
        input_squared = input.tensor ** 2
        
        # pre-allocate the output tensor
        output = torch.empty(b, c, *xyzo, device=input.tensor.device)
        
        # reshape the input to merge the spatial dimensions
        input = input.tensor.reshape(b, c, -1)
        
        # iterate through all field sizes
        for s, indices in self._indices.items():

            # compute the norms
            norms = input_squared[:, indices, :, :] \
                .view(b, -1, s, *xyzi) \
                .sum(dim=2)
            
            # run max-pooling on the norms-tensor
            _, indx = _MAX_POOLS[self.d](
                    norms,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.ceil_mode,
                    return_indices=True,
            )
            
            # in order to use the pooling indices computed for the norms to retrieve the fields, they need to be
            # expanded in the inner field dimension
            indx = indx.view(b, -1, 1, prod(xyzo)).expand(-1, -1, s, -1)

            # retrieve the fields from the input tensor using the pooling indeces
            output[:, indices, :, :] = input[:, indices, :] \
                .view(b, -1, s, prod(xyzi)) \
                .gather(3, indx) \
                .view(b, -1, *xyzo)
        
        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return calc_pool_output_shape(self.d, self, input_shape, self.out_type)
        
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test 
        # equivariance
        pass


class NormMaxPool2D(_NormMaxPoolND):
    
    def __init__(
            self,
            in_type: FieldType,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = None,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            ceil_mode: bool = False
    ):
        super().__init__(
                d=2,
                in_type=in_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
        )

class NormMaxPool3D(_NormMaxPoolND):
    
    def __init__(
            self,
            in_type: FieldType,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = None,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            ceil_mode: bool = False
    ):
        super().__init__(
                d=3,
                in_type=in_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
        )

# for backward compatibility
NormMaxPool = NormMaxPool2D
