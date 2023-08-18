from __future__ import annotations

import escnn.group

from .utils import *

from escnn.group import Group, GroupElement

from escnn.group import IrreducibleRepresentation
from escnn.group import Representation
from escnn.group import directsum
from escnn.group import utils

import numpy as np


from typing import Tuple, Callable, Iterable, List, Dict, Any, Union


__all__ = ["O2"]


class O2(Group):

    PARAM = 'radians'
    PARAMETRIZATIONS = [
        'radians',      # real in 0., 2pi/N, ... i*2pi/N, ...
        # 'C',            # point in the unit circle (i.e. cos and sin of 'radians')
        'MAT',  # 2x2 rotation matrix
    ]

    def __init__(self, maximum_frequency: int = 6):
        r"""
        Build an instance of the orthogonal group :math:`O(2)` which contains reflections and continuous planar
        rotations.
        
        Any group element is either a rotation :math:`r_{\theta}` by an angle :math:`\theta \in [0, 2\pi)` or a
        reflection :math:`f` followed by a rotation, i.e. :math:`r_{\theta}f`.
        Two reflections gives the identity :math:`f \cdot f = e` and a reflection commutes with a rotation by
        inverting it, i.e. :math:`r_\theta \cdot f = f \cdot r_{-\theta}`.
        A group element :math:`r_{\theta}f^j` is implemented as a pair :math:`(j, \theta)` with :math:`j \in \{0, 1\}`
        and :math:`\theta \in [0, 2\pi)`.
        
        .. note ::
        
            Since the group has infinitely many irreducible representations, it is not possible to build all of them.
            Each irrep is associated to one unique integer frequency and the parameter ``maximum_frequency`` specifies
            the maximum frequency of the irreps to build.
            New irreps (associated to higher frequencies) can be manually created by calling the method
            :meth:`~escnn.group.O2.irrep` (see the method's documentation).
        
        
        Subgroup Structure.
        
        A subgroup of :math:`O(2)` is identified by a tuple ``id`` :math:`(\theta, M)`.
        
        Here, :math:`M` can be either a positive integer indicating the number of rotations in the subgroup or
        :math:`-1`, indicating that the subgroup contains all continuous rotations.
        :math:`\theta` is either ``None`` or an angle in :math:`[0, \frac{2\pi}{M})`.
        If :math:`\theta` is ``None``, the subgroup does not contain any reflections.
        Otherwise, the subgroup contains the reflection :math:`r_{\theta}f` along the axis of the current group rotated
        by :math:`\frac{\theta}{2}`.
        
        Valid combinations are:
        
        - (``None``, :math:`M>0`): restrict to the cyclic subgroup :math:`C_M` generated by :math:`\langle r_{2\pi/M} \rangle`.
        
        - (``None``, :math:`-1`): restrict to the subgroup :math:`SO(2)` containing only the rotations
        
        - (:math:`\theta`, :math:`M>0`): restrict to the dihedral subgroup :math:`D_{M}` generated by :math:`\langle r_{2\pi/M}, r_{\theta} f \rangle`
        
        In particular:
        
        - (:math:`0.`, :math:`1`): restrict to the reflection group generated by :math:`\langle f \rangle`
        
        - (:math:`0.`, :math:`M`): restrict to the dihedral subgroup :math:`D_{M}` generated by :math:`\langle r_{2\pi/M}, f \rangle`
        
        - (``None``, :math:`1`): restrict to the cyclic subgroup of order 1 containing only the identity
        
        
        Args:
            maximum_frequency (int, optional): the maximum frequency to consider when building the irreps of the group
        
        Attributes:
            
            ~.reflection: the reflection element :math:`(j, \theta) = (1, 0.)`
            ~.rotation_order (int): this is equal to ``-1``, which means the group contains an infinite number of rotations

        """
        
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        
        super(O2, self).__init__("O(2)", True, False)
        
        self.rotation_order = -1
        
        self._maximum_frequency = maximum_frequency
        
        self._identity = self.element((0, 0.))
        self.reflection = self.element((1, 0.))
        
        self._build_representations()

    def __getinitargs__(self):
        return self._maximum_frequency,

    @property
    def generators(self) -> List[GroupElement]:
        raise ValueError(f'{self.name} is a continuous groups and '
                         f'some of its generators are infinitesimal. '
                         f'This is not currently supported')

    @property
    def identity(self) -> GroupElement:
        return self._identity

    @property
    def elements(self) -> List[GroupElement]:
        return None

    # @property
    # def elements_names(self) -> List[str]:
    #     return None

    @property
    def _keys(self) -> Dict[str, Any]:
        return dict()

    @property
    def subgroup_trivial_id(self):
        return (None, 1)

    @property
    def subgroup_self_id(self):
        return (0., -1)

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################
    
    def _inverse(self, element: Tuple[int, float], param: str = PARAM) -> Tuple[int, float]:
        r"""
        Return the inverse element of the input element.
        Given the element :math:`r_\theta f^j` as a pair :math:`(j, \theta)`,
        the method returns :math:`r_{-\theta}` (as :math:`(0, -\theta)`) if :math:`f = 0` and
        :math:`r_\theta f` (as :math:`(1, \theta)`) otherwise.
        
        Args:
            element (tuple): a group element :math:`r_{\theta}f^j` a pair :math:`(j, \theta)`

        Returns:
            its inverse
            
        """
        element = self._change_param(element, p_from=param, p_to='radians')
        inverse = element[0], -element[1] * (-1 if element[0] else 1)
        return self._change_param(inverse, p_from='radians', p_to=param)

    def _combine(self, e1: Tuple[int, float], e2: Tuple[int, float], param: str = PARAM, param1: str = None, param2: str = None) -> Tuple[int, float]:
        r"""
        Return the combination of the two input elements.
        Given two input element :math:`r_\alpha f^a` and :math:`r_\beta f^b`, the method returns
        :math:`r_\alpha f^a \cdot r_\beta f^b`.

        Args:
            e1 (tuple): a group element :math:`r_\alpha f^a` as a pair :math:`(a, \alpha)`
            e2 (tuple): another element :math:`r_\beta f^b` as a pair :math:`(b, \beta)`

        Returns:
            their combination :math:`r_\alpha f^a \cdot r_\beta f^b`
        """
        if param1 is None:
            param1 = param
        if param2 is None:
            param2 = param

        e1 = self._change_param(e1, p_from=param1, p_to='radians')
        e2 = self._change_param(e2, p_from=param2, p_to='radians')
        product = (e1[0] + e2[0]) % 2, e1[1] + (-1 if e1[0] else 1)*e2[1]
        return self._change_param(product, p_from='radians', p_to=param)

    def _equal(self, e1: Tuple[int, float], e2: Tuple[int, float], param: str = PARAM, param1: str = None, param2: str = None) -> bool:
        r"""

        Check if the two input values corresponds to the same element.

        See :meth:`escnn.group.SO2.equal` for more details.

        Args:
            e1 (tuple): an element
            e2 (tuple): another element
            
        Returns:
            whether they are the same element

        """
        if param1 is None:
            param1 = param
        if param2 is None:
            param2 = param

        e1 = self._change_param(e1, p_from=param1, p_to='radians')
        e2 = self._change_param(e2, p_from=param2, p_to='radians')
        return e1[0] == e2[0] and utils.cycle_isclose(e1[1], e2[1], 2 * np.pi)

    def _hash_element(self, element: Tuple[int, float], param: str = PARAM):
        element = self._change_param(element, p_from=param, p_to='radians')
        return hash(tuple(
            np.around(np.array([element[0], np.cos(element[1]), np.sin(element[1])]), 5)
        ))

    def _repr_element(self, element: Tuple[int, float], param: str = PARAM):
        element = self._change_param(element, p_from=param, p_to='radians')
        return "({}, {})".format('+' if not element[0] else '-', element[1])

    def _is_element(self, element: Tuple[int, float], param: str = PARAM, verbose: bool = False) -> bool:
        element = self._change_param(element, p_from=param, p_to='radians')
        if isinstance(element, tuple) and len(element) == 2 and isinstance(element[0], int) and isinstance(element[1], float):
            return element[0] in {0, 1}
        else:
            return False

    def _change_param(self, element: Tuple[int, float], p_from: str, p_to: str):
        assert p_from in self.PARAMETRIZATIONS
        assert p_to in self.PARAMETRIZATIONS

        flip, rotation = element

        # convert to radians
        if p_from == 'MAT':
            assert isinstance(rotation, np.ndarray)
            assert rotation.shape == (2, 2)
            assert np.isclose(np.linalg.det(rotation), 1.)
            assert np.allclose(rotation @ rotation.T, np.eye(2), atol=1e-6), np.fabs(rotation @ rotation.T - np.eye(2))
            assert np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-6), np.fabs(rotation.T @ rotation - np.eye(2))

            cos = (rotation[0, 0] + rotation[1, 1]) / 2.
            sin = (rotation[1, 0] - rotation[0, 1]) / 2.
    
            rotation = np.arctan2(sin, cos)
            
        elif p_from == 'radians':
            assert isinstance(rotation, float)
        else:
            raise ValueError('Parametrization {} not recognized'.format(p_from))

        rotation = rotation % (2*np.pi)
        
        # convert from radians
        if p_to == 'radians':
            pass
        elif p_to == 'MAT':
            cos = np.cos(rotation)
            sin = np.sin(rotation)
            rotation = np.array(([
                [cos, -sin],
                [sin, cos],
            ]))
        else:
            raise ValueError('Parametrization {} not recognized'.format(p_to))
        
        return flip, rotation

    ###########################################################################
    
    def sample(self) -> GroupElement:
        return self.element((np.random.randint(0, 2), np.random.random() * 2 * np.pi))

    def grid(self, N: int, type: str = 'regular', seed: int = None) -> List[GroupElement]:
        r"""
            .. todo::
                Add documentation

        """

        if type == 'regular':
            grid = [
                (0, i * 2*np.pi / N) for i in range(N)
            ] + [
                (1, i * 2 * np.pi / N) for i in range(N)
            ]
        elif type == 'rand':
            if isinstance(seed, int):
                rng = np.random.RandomState(seed)
            elif seed is None:
                rng = np.random
            else:
                assert isinstance(seed, np.random.RandomState)
                rng = seed
            grid = [(rng.randint(0, 2), rng.random()*2*np.pi) for i in range(N)]
        else:
            raise ValueError(f'Grid type {type} not recognized')

        return [
            self.element(g, param='radians')
            for g in grid
        ]

    def grid_so2(self, N: int, type: str = 'regular', seed: int = None) -> List[GroupElement]:
        r"""
            .. todo::
                Add documentation

        """

        if type == 'regular':
            so2_grid = [i * 2*np.pi / N for i in range(N)]
        elif type == 'rand':
            if isinstance(seed, int):
                rng = np.random.RandomState(seed)
            elif seed is None:
                rng = np.random
            else:
                assert isinstance(seed, np.random.RandomState)
                rng = seed
            so2_grid = [rng.random()*2*np.pi for i in range(N)]
        else:
            raise ValueError(f'Grid type {type} not recognized')

        return [self.element((0, g), param='radians') for g in so2_grid]

    def testing_elements(self, n=4*13) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        """
        return iter(
            [self.element((0, i * 2. * np.pi / n)) for i in range(n)]
          + [self.element((1, i * 2. * np.pi / n)) for i in range(n)]
        )
    
    def __eq__(self, other):
        if not isinstance(other, O2):
            return False
        else:
            return self.name == other.name # and self._maximum_frequency == other._maximum_frequency

    def _subgroup(self, id: Tuple[float, int]) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement]
    ]:
        r"""
        Restrict the current group :math:`O(2)` to the subgroup identified by the input ``id``, where ``id`` is a
        tuple :math:`(\theta, M)`.
        
        Args:
            id (tuple): the identification of the subgroup

        Returns:
            a tuple containing
            
                - the subgroup
                
                - a function which maps an element of the subgroup to its inclusion in the original group and
                
                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
                
        """
        
        assert isinstance(id, tuple) and len(id) == 2, id
        assert id[0] is None or isinstance(id[0], float), id[0]
        assert isinstance(id[1], int), id[1]

        order = id[1]
        axis = id[0]
        # assert (id[0] is None and (id[1] > 0 or id[1] == -1)) or (id[0] is not None and id[1] > 0)
        # assert axis is None or 0 <= axis < 2*np.pi/order or order < 0, (axis, order)
        assert axis is None or 0 <= axis <= 2*np.pi, (axis, order)
        assert order == -1 or order > 0, (axis, order)
        
        if id[0] is not None and id[1] == -1:
            sg = escnn.group.o2_group(self._maximum_frequency)
            parent_mapping = build_adjoint_map(self, self.element((0, axis / 2)))
            child_mapping = build_adjoint_map(self, self.element((0, -axis / 2)))

        elif id[0] is None and id[1] == -1:
            sg = escnn.group.so2_group(self._maximum_frequency)
            # parent_mapping = lambda e: self.element((0, e._element))
            # child_mapping = lambda e, sg=sg: None if e._element[0] != 0 else sg.element(e._element[1])
            parent_mapping = so2_to_o2(self)
            child_mapping = o2_to_so2(sg)

        elif id[0] is not None and id[1] == 1:
            # take the elements of the group generated by "2pi/k f"
            sg = escnn.group.cyclic_group(2)
            # parent_mapping = lambda e, axis=axis: self.element((e._element, axis * e._element))
            # child_mapping = lambda e, axis=axis, sg=sg: None if not utils.cycle_isclose(e._element[1], axis * e._element[0], 2 * np.pi) else sg.element(e._element[0])
            parent_mapping = flip_to_o2(axis, self)
            child_mapping = o2_to_flip(axis, sg)

        elif id[0] is None:
            # take the elements of the group generated by "2pi/order"
            sg = escnn.group.cyclic_group(order)
            # parent_mapping = lambda e, order=order: self.element((0, e._element * 2. * np.pi / order))
            # child_mapping = lambda e, order=order, sg=sg: None if (e._element[0] != 0 or not utils.cycle_isclose(e._element[1], 0., 2. * np.pi / order)) else \
            #     sg.element(int(round(e._element[1] * order / (2. * np.pi))))
            parent_mapping = so2_to_o2(self)
            child_mapping = o2_to_so2(sg)

        elif id[0] is not None and id[1] > 1:
            # take the elements of the group generated by "2pi/order" and "2pi/k f"
            sg = escnn.group.dihedral_group(order)

            # parent_mapping = lambda e, order=order, axis=axis: self.element((e._element[0], e._element[1] * 2. * np.pi / order + e._element[0] * axis))
            # child_mapping = lambda e, order=order, axis=axis, sg=sg: None if not utils.cycle_isclose(e._element[1] - e._element[0] * axis, 0., 2. * np.pi / order) else \
            #     sg.element((e._element[0], int(round((e._element[1] - e._element[0] * axis) * order / (2. * np.pi)))))
            parent_mapping = dn_to_o2(axis, self)
            child_mapping = o2_to_dn(axis, sg)
        else:
            raise ValueError(f"id '{id}' not recognized")

        return sg, parent_mapping, child_mapping

    def _combine_subgroups(self, sg_id1, sg_id2):
    
        sg_id1 = self._process_subgroup_id(sg_id1)
        sg1, inclusion, restriction = self.subgroup(sg_id1)
        sg_id2 = sg1._process_subgroup_id(sg_id2)
    
        if sg_id1[0] is None:
            return (None, sg_id2)
        elif sg_id1[1] == 1:
            return sg_id1[0] if sg_id2 == 2 else None, 1
        elif sg_id2[0] is None:
            return sg_id2
        else:
            flip = sg_id1[0] + inclusion(sg1.element((0, sg_id2[0]))).to('radians')[1]
            return (flip,) + sg_id2[1:]

    def _restrict_irrep(self, irrep: Tuple, id: Tuple[int, int]) -> Tuple[np.matrix, List[Tuple]]:
        r"""
        Restrict the input irrep of current group to the subgroup identified by "id".
        More precisely, "id" is a tuple :math:`(k, m)`, where :math:`m` is a positive integer indicating the number of
        rotations in the subgroup while :math:`k` is either None (no flips in the subgroup) or an angle in
        :math:`[0, \frac{2\pi}{m})` (indicating the axis of flip in the subgroup).
        Valid combinations are:
        - (None, -1): restrict to the subgroup :math:`SO(2)` containing only the rotations
        - (None, m): restrict to the cyclic subgroup with order "m" :math:`C_m` generated by :math:`\langle r_{2\pi/m} \rangle`.
        - (0, m): restrict to the dihedral subgroup with order "2m" :math:`D_{2m}` generated by :math:`\langle r_{2\pi/m}, f \rangle`
        - (0, 1): restrict to the cyclic subgroup of order 2 :math:`C_2` generated by the flip :math:`\langle f \rangle`
        - (None, 1): restrict to the cyclic subgroup of order 1 :math:`C_1` containing only the identity
        - (\theta, m): restrict to the dihedral subgroup with order "2m" :math:`D_{2m}` generated by :math:`\langle r_{2\pi/m}, r_{\theta} f \rangle`
        
        Args:
            irrep (tuple): the identifier of the irrep to restrict
            id (tuple): the identification of the subgroup

        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """

        irr = self.irrep(*irrep)
        
        sg, _, _ = self.subgroup(id)

        irreps = []
        change_of_basis = None

        if id[0] is not None and id[1] == -1:
            
            change_of_basis = np.eye(irr.size)
            irreps.append(irr.id)

            if irr.size == 2:
                f = irr.attributes["frequency"]
                change_of_basis = utils.psi(0.5 * id[0], f) @ change_of_basis

        elif id[0] is None and id[1] == -1:
            f = irr.attributes["frequency"]
            irreps.append((f,))
            change_of_basis = np.eye(irr.size)
            
        elif id[0] is not None and id[1] == 1:
            j = irr.attributes["flip_frequency"]
            k = irr.attributes["frequency"]
            change_of_basis = np.eye(irr.size)
            if irr.size > 1:
                irreps.append((0,))
                change_of_basis = utils.psi(0.5 * id[0], k)
            irreps.append((j,))
        elif id[0] is None and id[1] > 0:
        
            order = id[1]

            f = irr.attributes["frequency"] % order
            if f > order/2:
                f = order - f
                change_of_basis = utils.chi(1)
            else:
                change_of_basis = np.eye(irr.size)

            r = (f,)
            if sg.irrep(*r).size < irr.size:
                irreps.append(r)
            irreps.append(r)
    
        elif id[0] is not None and id[1] > 1:
        
            order = id[1]
            j = irr.attributes["flip_frequency"]
            f = irr.attributes["frequency"]
            k = f % order
            
            if k > order/2:
                k = order - k
                change_of_basis = np.array([[1, 0], [0, -1]])
            else:
                change_of_basis = np.eye(irr.size)
            
            r = (j,k)
            if sg.irrep(*r).size < irr.size:
                irreps.append((0, k))
            irreps.append(r)
                
            if irr.size == 2:
                change_of_basis = utils.psi(0.5 * id[0], f) @ change_of_basis

        else:
            raise ValueError(f"id '{id}' not recognized")
        
        return change_of_basis, irreps

    def _build_representations(self):
        r"""
        Build the irreps for this group

        """
        
        # Build all the Irreducible Representations
    
        j, k = 0, 0
    
        # add Trivial representation
        self.irrep(j, k)
    
        j = 1
        for k in range(self._maximum_frequency + 1):
            self.irrep(j, k)

        # Build all Representations
        
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name : irr for irr in self.irreps()})

    def bl_regular_representation(self, L: int) -> Representation:
        r"""
        Band-Limited regular representation up to frequency ``L`` (included).

        Args:
            L(int): max frequency

        """
        name = f'regular_{L}'
    
        if name not in self._representations:
            irreps = []
        
            for l in range(L + 1):
                if l == 0:
                    irreps += [self.irrep(0, l)]

                irreps += [self.irrep(1, l)] * self.irrep(1, l).size
        
            self._representations[name] = directsum(irreps, name=name)
    
        return self._representations[name]

    def bl_quotient_representation(self,
                                   L: int,
                                   subgroup_id,
                                   name: str = None,
                                   ) -> escnn.group.Representation:
        r"""
        Band-Limited quotient representation up to frequency ``L`` (included).

        The quotient representation corresponds to the action of the current group :math:`G` on functions over the
        homogeneous space :math:`G/H`, where :math:`H` is the subgroup of :math:`G` identified by ``subgroup_id``.

        Args:
            L(int): max frequency
            subgroup_id: id identifying the subgroup H.
            name (str, optional)

        """

        if name is None:
            name = f"quotient[{subgroup_id}]_{L}"

        if name not in self.representations:
            subgroup, _, _ = self.subgroup(subgroup_id)

            homspace = self.homspace(subgroup_id)

            irreps = []

            irreps_ids = [(0,0)] + [(1, l) for l in range(L+1)]

            for id in irreps_ids:
                irr = self.irrep(*id)
                multiplicity = homspace.dimension_basis(irr.id, homspace.H.trivial_representation.id)[1]
                irreps += [irr] * multiplicity

            self.representations[name] = directsum(irreps, name=name)

        return self.representations[name]

    def bl_irreps(self, L: int) -> List[Tuple]:
        r"""
        Returns a list containing the id of all irreps of (rotational) frequency smaller or equal to ``L``.
        This method is useful to easily specify the irreps to be used to instantiate certain objects, e.g. the
        Fourier based non-linearity :class:`~escnn.nn.FourierPointwise`.
        """
        assert 0 <= L, L
        irreps = [(0, 0)]
        for l in range(L + 1):
            irreps += [(1, l)]
        return irreps

    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0,0']

    def standard_representation(self) -> Representation:
        r"""
        Standard representation of :math:`\O2` as 2x2 rotation matrices

        This is equivalent to ``self.irrep(1, 1)``.

        """
        return self.irrep(1, 1)

    def irrep(self, j: int, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with reflection and rotation frequencies :math:`j` (reflection) and :math:`k` (rotation) of the
        current group.
        Notice: the frequencies has to be non-negative integers: :math:`j \in \{0, 1\}` and :math:`k \in \mathbb{N}`
        
        Valid parameters are :math:`(0, 0)` and :math:`(1, 0)`, :math:`(1, 1)`, :math:`(1, 2)`, :math:`(1, 3)`, ...
        
        Args:
            j (int): the frequency of the reflection in the irrep
            k (int): the frequency of the rotations in the irrep

        Returns:
            the corresponding irrep

        """
    
        assert j in [0, 1]
        assert k >= 0
    
        name = f"irrep_{j},{k}"
        id = (j, k)

        if id not in self._irreps:
            irrep = _build_irrep_o2(j, k)
            character = _build_char_o2(j, k)
            
            if j == 0:
                if k == 0:
                    # Trivial representation
                    supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                    self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                                  supported_nonlinearities=supported_nonlinearities,
                                                                  character=character,
                                                                  # trivial=True,
                                                                  frequency=k,
                                                                  flip_frequency=j
                                                                  )
                else:
                    raise ValueError(f"Error! Flip frequency {j} and rotational frequency {k} don't correspond to any irrep of the group {self.name}!")
                
            elif k == 0:

                # add Trivial on SO(2) subgroup Representation
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k,
                                                              flip_frequency=j
                                                              )
            else:
                # 2 dimensional Irreducible Representations
                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 2, 'R',
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k,
                                                              flip_frequency=j
                                                              )

        return self._irreps[id]

    def _induced_from_irrep(self,
                            subgroup_id: Tuple[float, int],
                            repr: IrreducibleRepresentation,
                            representatives: List[GroupElement] = None,
                            ) -> Tuple[List[IrreducibleRepresentation], np.ndarray, np.ndarray]:
        
        if representatives is None:
            if subgroup_id == (None, -1):
                # SO(2) has finite index in O(2)
                # so we can build the set of representatives explicitly
                representatives = [self.identity, self.reflection]
            else:
                raise ValueError(f"Induction from discrete subgroups of O(2) leads to infinite dimensional induced "
                                 f"representations. Hence, induction from the subgroup identified "
                                 f"by {subgroup_id} is not allowed.")
        
        return super(O2, self)._induced_from_irrep(subgroup_id, repr, representatives)

    _cached_group_instance = None

    @classmethod
    def _generator(cls, maximum_frequency: int = 3) -> 'O2':
        if cls._cached_group_instance is None:
            cls._cached_group_instance = O2(maximum_frequency)
        elif cls._cached_group_instance._maximum_frequency < maximum_frequency:
            cls._cached_group_instance._maximum_frequency = maximum_frequency
            cls._cached_group_instance._build_representations()
    
        return cls._cached_group_instance


def _build_irrep_o2(j: int, k: int):
    
    def irrep(element: GroupElement, j=j, k=k) -> np.ndarray:
        assert j in [0, 1]
        assert k >= 0
        
        if j == 0:
            if k == 0:
                # Trivial representation
                return np.eye(1)
            else:
                raise ValueError(
                    f"Error! Flip frequency {j} and rotational frequency {k} don't correspond to any irrep of the group {element.group.name}!")
        elif k == 0:
            # Trivial on SO(2) subgroup Representation
            return np.array([[-1 if element.to('radians')[0] else 1]])
        else:
            e = element.to('radians')
            # 2 dimensional Irreducible Representations
            return utils.psichi(e[1], e[0], k=k)
        
    return irrep


def _build_char_o2(j: int, k: int):
    
    def character(element: GroupElement, j=j, k=k) -> float:
        assert j in [0, 1]
        assert k >= 0
        
        if j == 0:
            if k == 0:
                # Trivial representation
                return 1.
            else:
                raise ValueError(
                    f"Error! Flip frequency {j} and rotational frequency {k} don't correspond to any irrep of the group {element.group.name}!")
        elif k == 0:
            # add Trivial on SO(2) subgroup Representation
            return -1 if element.to('radians')[0] else 1
        else:
            e = element.to('radians')
            # 2 dimensional Irreducible Representations
            return 0 if e[0] else (2 * np.cos(k * e[1]))
        
    return character


# SO2 (and Cyclic) ###############################

def o2_to_so2(so2: Union[escnn.group.SO2, escnn.group.CyclicGroup]):

    def _map(e: GroupElement, so2=so2):
        assert isinstance(e.group, O2)

        flip, rotation = e.to('radians')

        if flip == 0:
            try:
                return so2.element(rotation, 'radians')
            except ValueError:
                return None
        else:
            return None

    return _map


def so2_to_o2(o2: O2):

    def _map(e: GroupElement, o2=o2):
        assert isinstance(e.group, escnn.group.SO2) or isinstance(e.group, escnn.group.CyclicGroup)
        return o2.element(
            (0, e.to('radians')), 'radians'
        )

    return _map


# Flip wrt an axis ######################################


def o2_to_flip(axis: float, flip: escnn.group.CyclicGroup):
    assert isinstance(flip, escnn.group.CyclicGroup) and flip.order() == 2

    def _map(e: GroupElement, flip=flip, axis=axis):
        assert isinstance(e.group, O2)

        f, rot = e.to('radians')

        if f == 0 and utils.cycle_isclose(rot, 0, 2*np.pi):
            return flip.identity
        elif f == 1 and utils.cycle_isclose(rot, axis, 2*np.pi):
            return flip.element(1)
        else:
            return None

    return _map


def flip_to_o2(axis: float, o2: O2):

    def _map(e: GroupElement, axis=axis, o2=o2):
        assert isinstance(e.group, escnn.group.CyclicGroup) and e.group.order() == 2

        f = e.to('int')

        if f == 0:
            return o2.identity
        else:
            return o2.element((1, axis))

    return _map


# Dihedral Group ######################################


def o2_to_dn(axis: float, dn: escnn.group.DihedralGroup):
    assert isinstance(dn, escnn.group.DihedralGroup)

    def _map(e: GroupElement, dn=dn, axis=axis):
        assert isinstance(e.group, O2)

        f, rot = e.to('radians')

        if utils.cycle_isclose(rot - f * axis, 0., 2. * np.pi / dn.rotation_order):
            return dn.element((f, int(round((rot - f * axis) * dn.rotation_order / (2. * np.pi)))))
        else:
            return None

    return _map


def dn_to_o2(axis: float, o2: O2):

    def _map(e: GroupElement, axis=axis, o2=o2):
        assert isinstance(e.group, escnn.group.DihedralGroup)

        f, rot = e.to('int')

        return o2.element((f, rot * 2. * np.pi / e.group.rotation_order + f * axis))

    return _map



