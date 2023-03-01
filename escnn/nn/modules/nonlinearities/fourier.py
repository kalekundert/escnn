
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any

import numpy as np

__all__ = ["FourierPointwise", "FourierELU"]


def _build_kernel(G: Group, irrep: List[tuple]):
    kernel = []
    
    for irr in irrep:
        irr = G.irrep(*irr)
        
        c = int(irr.size//irr.sum_of_squares_constituents)
        k = irr(G.identity)[:, :c] * np.sqrt(irr.size)
        kernel.append(k.T.reshape(-1))
    
    kernel = np.concatenate(kernel)
    return kernel
    

class FourierPointwise(EquivariantModule):
    
    def __init__(
            self,
            gspace: GSpace,
            channels: int,
            irreps: List,
            *grid_args,
            function: str = 'p_relu',
            inplace: bool = True,
            out_irreps: List = None,
            **grid_kwargs
    ):
        r"""
        
        Applies a Inverse Fourier Transform to sample the input features, apply the pointwise non-linearity in the
        group domain (Dirac-delta basis) and, finally, computes the Fourier Transform to obtain irreps coefficients.
        
        .. warning::
            This operation is only *approximately* equivariant and its equivariance depends on the sampling grid and the
            non-linear activation used, as well as the original band-limitation of the input features.
            
        The same function is applied to every channel independently.
        The input representation is preserved by this operation and, therefore, it equals the output representation.
        
        The class first constructs a band-limited regular representation of ```gspace.fibergroup``` using
        :meth:`escnn.group.Group.spectral_regular_representation`.
        The band-limitation of this representation is specified by ```irreps``` which should be a list containing
        a list of ids identifying irreps of ```gspace.fibergroup``` (see :attr:`escnn.group.IrreducibleRepresentation.id`).
        This representation is used to define the input and output field types, each containing ```channels``` copies
        of a feature field transforming according to this representation.
        A feature vector transforming according to such representation is interpreted as a vector of coefficients
        parameterizing a function over the group using a band-limited Fourier basis.

        .. note::
            Instead of building the list ``irreps`` manually, most groups implement a method ``bl_irreps()`` which can be
            used to generate this list with through a simpler interface. Check each group's documentation.
        
        To approximate the Fourier transform, this module uses a finite number of samples from the group.
        The set of samples used is specified by the ```grid_args``` and ```grid_kwargs``` which are forwarded to
        the method :meth:`~escnn.group.Group.grid`.
        
        .. todo::
            Mention the normalization of the transform we use
        
        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            function (str): the identifier of the non-linearity.
                    It is used to specify which function to apply.
                    By default (``'p_relu'``), ReLU is used.
            *grid_args: parameters used to construct the discretization grid
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            **grid_kwargs: keyword parameters used to construct the discretization grid
            
        """

        assert isinstance(gspace, GSpace)
        
        super(FourierPointwise, self).__init__()

        self.space = gspace
        
        G: Group = gspace.fibergroup
        
        self.rho = G.spectral_regular_representation(*irreps, name=None)

        self.in_type = FieldType(self.space, [self.rho] * channels)

        if out_irreps is None:
            # the representation in input is preserved
            self.out_type = self.in_type
            self.rho_out = self.rho
        else:
            self.rho_out = G.spectral_regular_representation(*out_irreps, name=None)
            self.out_type = FieldType(self.space, [self.rho_out] * channels)

        # retrieve the activation function to apply
        if function == 'p_relu':
            self._function = F.relu_ if inplace else F.relu
        elif function == 'p_elu':
            self._function = F.elu_ if inplace else F.elu
        elif function == 'p_sigmoid':
            self._function = torch.sigmoid_ if inplace else F.sigmoid
        elif function == 'p_tanh':
            self._function = torch.tanh_ if inplace else F.tanh
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        
        kernel = _build_kernel(G, irreps)
        assert kernel.shape[0] == self.rho.size
        
        kernel = kernel / np.linalg.norm(kernel)
        kernel = kernel.reshape(-1, 1)
        
        grid = G.grid(*grid_args, **grid_kwargs)
        
        A = np.concatenate(
            [
                self.rho(g) @ kernel
                for g in grid
            ], axis=1
        ).T

        if out_irreps is not None:

            _missing_input_irreps = list(set(irreps).difference(set(out_irreps)))
            rho_out_extended = G.spectral_regular_representation(*out_irreps, *_missing_input_irreps, name=None)
            kernel_out = _build_kernel(G, out_irreps + _missing_input_irreps)
            assert kernel_out.shape[0] == rho_out_extended.size

            kernel_out = kernel_out / np.linalg.norm(kernel_out)
            kernel_out = kernel_out.reshape(-1, 1)

            A_out = np.concatenate(
                [
                    rho_out_extended(g) @ kernel_out
                    for g in grid
                ], axis=1
            ).T
        else:
            A_out = A
            _missing_input_irreps = []
            rho_out_extended = self.rho_out

        eps = 1e-8
        Ainv = np.linalg.inv(A_out.T @ A_out + eps * np.eye(rho_out_extended.size)) @ A_out.T

        if out_irreps is not None:
            Ainv = Ainv[:self.rho_out.size, :]

        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))
        self.register_buffer('Ainv', torch.tensor(Ainv, dtype=torch.get_default_dtype()))
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies the pointwise activation function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after the non-linearities have been applied

        """

        assert input.type == self.in_type
        
        shape = input.shape
        x_hat = input.tensor.view(shape[0], len(self.in_type), self.rho.size, *shape[2:])
        
        x = torch.einsum('bcf...,gf->bcg...', x_hat, self.A)
        
        y = self._function(x)

        y_hat = torch.einsum('bcg...,fg->bcf...', y, self.Ainv)

        y_hat = y_hat.reshape(shape[0], self.out_type.size, *shape[2:])

        return GeometricTensor(y_hat, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, atol: float = 1e-5, rtol: float = 2e-2) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size
        B = 128
        x = torch.randn(B, c, *[3]*self.space.dimensionality)

        # since we mostly use non-linearities like relu or elu, we make sure the average value of the features is
        # positive, such that, when we test inputs with only frequency 0 (or only low frequencies), the output is not
        # zero everywhere
        x = x.view(B, len(self.in_type), self.rho.size, *[3]*self.space.dimensionality)
        p = 0
        for irr in self.rho.irreps:
            irr = self.space.irrep(*irr)
            if irr.is_trivial():
                x[:, :, p] = x[:, :, p].abs()
            p+=irr.size

        x = x.view(B, self.in_type.size, *[3]*self.space.dimensionality)

        errors = []

        # for el in self.space.testing_elements:
        for _ in range(100):
            
            el = self.space.fibergroup.sample()
    
            x1 = GeometricTensor(x.clone(), self.in_type)
            x2 = GeometricTensor(x.clone(), self.in_type).transform_fibers(el)

            out1 = self(x1).transform_fibers(el)
            out2 = self(x2)

            out1 = out1.tensor.view(B, len(self.out_type), self.rho_out.size, *out1.shape[2:]).detach().numpy()
            out2 = out2.tensor.view(B, len(self.out_type), self.rho_out.size, *out2.shape[2:]).detach().numpy()

            errs = np.linalg.norm(out1 - out2, axis=2).reshape(-1)
            errs[errs < atol] = 0.
            norm = np.sqrt(np.linalg.norm(out1, axis=2).reshape(-1) * np.linalg.norm(out2, axis=2).reshape(-1))
            
            relerr = errs / norm

            # print(el, errs.max(), errs.mean(), relerr.max(), relerr.min())
        
            assert relerr.mean()+ relerr.std() < rtol, \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {}, std ={}' \
                    .format(el, relerr.max(), relerr.mean(), relerr.std())
            errors.append((el, errs.mean()))

        return errors


class FourierELU(FourierPointwise):
    
    def __init__(
            self,
            gspace: GSpace, channels: int, irreps: List, *grid_args,
            inplace: bool = True, out_irreps: List = None,  **grid_kwargs
    ):
        r"""

        Applies a Inverse Fourier Transform to sample the input features, apply ELU point-wise in the
        group domain (Dirac-delta basis) and, finally, computes the Fourier Transform to obtain irreps coefficients.
        See :class:`~escnn.nn.FourierPointwise` for more details.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            **grid_kwargs: keyword parameters used to construct the discretization grid

        """
        
        super(FourierELU, self).__init__(
            gspace, channels, irreps, *grid_args,
            function='p_elu', inplace=inplace, out_irreps=out_irreps, **grid_kwargs)

