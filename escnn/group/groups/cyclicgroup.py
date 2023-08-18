from __future__ import annotations

import escnn.group
from escnn.group import Group, GroupElement
from escnn.group import IrreducibleRepresentation, Representation
from escnn.group import utils

from .utils import *

import numpy as np
import math

from typing import Tuple, Callable, Iterable, List, Dict, Any


__all__ = ["CyclicGroup"]


class CyclicGroup(Group):

    PARAM = 'int'
    PARAMETRIZATIONS = [
        'int',          # integer in 0, 1, ..., N-1
        'radians',      # real in 0., 2pi/N, ... i*2pi/N, ...
        # 'C',            # point in the unit circle (i.e. cos and sin of 'radians')
        'MAT',          # 2x2 rotation matrix
    ]

    def __init__(self, N: int):
        r"""
        Build an instance of the cyclic group :math:`C_N` which contains :math:`N` discrete planar rotations.
        
        The group elements are :math:`\{e, r, r^2, r^3, \dots, r^{N-1}\}`, with group law
        :math:`r^a \cdot r^b = r^{\ a + b \!\! \mod \!\! N \ }`.
        The cyclic group :math:`C_N` is isomorphic to the integers *modulo* ``N``.
        For this reason, elements are stored as the integers between :math:`0` and :math:`N-1`, where the :math:`k`-th
        element can also be interpreted as the discrete rotation by :math:`k\frac{2\pi}{N}`.
        
        Subgroup Structure.
        
        A subgroup of :math:`C_N` is another cyclic group :math:`C_M` and is identified by an ``id`` containing the
        integer :math:`M` (i.e. the order of the subgroup).
        
        If the current group is :math:`C_N`, the subgroup is generated by :math:`r^{(N/M)}`.
        Notice that :math:`M` has to divide the order :math:`N` of the group.
        
        Args:
            N (int): order of the group

        Attributes:
            ~.rotation_order (int): the number of rotations, i.e. the order of the group

        """
        
        assert (isinstance(N, int) and N > 0), N
        
        super(CyclicGroup, self).__init__("C%d" % N, False, True)
        
        self.N = N

        # int: for consistency with the DihedralGroup, store the number of rotations also in this attribute
        self.rotation_order = N

        self._elements = [self.element(i) for i in range(N)]
        # self._elements_names = ['e'] + ['r%d' % i for i in range(1, N)]

        self._identity = self.element(0)
        
        self._build_representations()

    def __getinitargs__(self):
        return self.N,

    @property
    def generators(self) -> List[GroupElement]:
        if self.order() > 1:
            return [self.element(1)]
        else:
            # the generator of the trivial group is the empty set
            return []

    @property
    def identity(self) -> GroupElement:
        return self._identity

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    # @property
    # def elements_names(self) -> List[str]:
    #     return self._elements_names

    @property
    def _keys(self) -> Dict[str, Any]:
        return {'N': self.N}

    @property
    def subgroup_trivial_id(self):
        return 1

    @property
    def subgroup_self_id(self):
        return self.order()

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################

    def _inverse(self, element: int, param: str = PARAM) -> int:
        r"""
        Return the inverse element :math:`r^{-j \mod N}` of the input element :math:`r^j`, specified by the input
        integer :math:`j` (``element``)
        
        Args:
            element (int): a group element :math:`r^j`

        Returns:
            its opposite :math:`r^{-j \mod N}`
            
        """
        self._change_param(element, param, 'int')
        element = (-element) % self.N
        return self._change_param(element, 'int', param)

    def _combine(self, e1: int, e2: int, param: str = PARAM, param1: str = None, param2: str = None) -> int:
        r"""
        Return the composition of the two input elements.
        Given two integers :math:`a` and :math:`b` representing the elements :math:`r^a` and :math:`r^b`, the method
        returns the integer :math:`a + b \mod N` representing the element :math:`r^{a + b \mod N}`.
        

        Args:
            e1 (int): a group element :math:`r^a`
            e2 (int): another group element :math:`r^a`

        Returns:
            their composition :math:`r^{a+b \mod N}`
            
        """

        if param1 is None:
            param1 = param
        if param2 is None:
            param2 = param

        e1 = self._change_param(e1, p_from=param1, p_to='int')
        e2 = self._change_param(e2, p_from=param2, p_to='int')
        return self._change_param(
            (e1 + e2) % self.N,
            p_from = 'int',
            p_to = param
        )

    def _equal(self, e1: int, e2: int, param: str = PARAM, param1: str = None, param2: str = None) -> bool:
        r"""

        Check if the two input values corresponds to the same element.

        Args:
            e1 (int): an element
            e2 (int): another element

        Returns:
            whether they are the same element

        """
        if param1 is None:
            param1 = param
        if param2 is None:
            param2 = param

        e1 = self._change_param(e1, p_from=param1, p_to='int')
        e2 = self._change_param(e2, p_from=param2, p_to='int')
        return e1 == e2
    
    def _is_element(self, element: int, param: str = PARAM, verbose: bool = False) -> bool:
    
        element = self._change_param(element, p_from=param, p_to='int')
        if isinstance(element, int):
            return 0 <= element < self.N
        else:
            return False

    def _hash_element(self, element: int, param: str = PARAM):
        element = self._change_param(element, p_from=param, p_to='int')
        return hash(element)

    def _repr_element(self, element: int, param: str = PARAM):
        element = self._change_param(element, p_from=param, p_to='int')
        return "{}[2pi/{}]".format(element, self.N)
    
    def _change_param(self, element, p_from: str, p_to: str):
        assert p_from in self.PARAMETRIZATIONS
        assert p_to in self.PARAMETRIZATIONS
        
        if p_from == 'MAT':
            assert isinstance(element, np.ndarray)
            assert element.shape == (2, 2)
            assert np.isclose(np.linalg.det(element), 1.)
            assert np.allclose(element @ element.T, np.eye(2))
    
            cos = (element[0, 0] + element[1, 1]) / 2.
            sin = (element[1, 0] - element[1, 0]) / 2.
    
            element = np.arctan2(sin, cos)
            p_from = 'radians'

        # convert to INT
        if p_from == 'int':
            assert isinstance(element, int), element
        elif p_from == 'radians':
            assert isinstance(element, float), element
            if not utils.cycle_isclose(element, 0., 2*np.pi/self.N):
                raise ValueError()
            element = int(round(self.N * element / (2*np.pi))) % self.N
        else:
            raise ValueError('Parametrization {} not recognized'.format(p_from))

        # convert from INT
        if p_to == 'int':
            return element
        elif p_to == 'radians':
            return element * (2*np.pi) / self.N
        elif p_to == 'MAT':
            element = element * (2*np.pi) / self.N
            cos = np.cos(element)
            sin = np.sin(element)
            return np.array(([
                [cos, -sin],
                [sin, cos],
            ]))
        else:
            raise ValueError('Parametrization {} not recognized'.format(p_to))

    ###########################################################################

    def sample(self) -> GroupElement:
        return self.element(np.random.randint(0, self.order()))

    def testing_elements(self) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.
        
        """
        return iter(self._elements)

    def __eq__(self, other):
        if not isinstance(other, CyclicGroup):
            return False
        else:
            return self.name == other.name and self.order() == other.order()

    def _subgroup(self, id: int) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement]
    ]:
        r"""
        Restrict the current group to the cyclic subgroup :math:`C_M`.
        If the current group is :math:`C_N`, it restricts to the subgroup generated by :math:`r^{(N/M)}`.
        Notice that :math:`M` has to divide the order :math:`N` of the current group.
        
        The method takes as input the integer :math:`M` identifying of the subgroup to build (the order of the subgroup)
        
        Args:
            id (int): the integer :math:`M` identifying of the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
                
        """

        assert isinstance(id, int), id

        order = id

        assert self.order() % order == 0, \
            "Error! The subgroups of a cyclic group have an order that divides the order of the supergroup." \
            " %d does not divide %d " % (order, self.order())

        # Build the subgroup
        
        # take the elements of the group generated by "r^ratio"
        sg = escnn.group.cyclic_group(order)

        # parent_mapping = lambda e, ratio=ratio: self.element(e._element * ratio)
        # child_mapping = lambda e, ratio=ratio, sg=sg: None if e._element % ratio != 0 else sg.element(int(e._element // ratio))
        parent_mapping = _build_parent_map(self, order)
        child_mapping = _build_child_map(self, sg)

        return sg, parent_mapping, child_mapping

    def grid(self, type: str, N: int) -> List[GroupElement]:
        r"""
            .. todo ::
                Add docs

        """

        if type == 'rand':
            return [self.sample() for _ in range(N)]
        elif type == 'regular':
            assert self.order() % N == 0
            r = self.order() // N
            return [self.element(i*r) for i in range(N)]
        else:
            raise ValueError(f'Grid type "{type}" not recognized!')

    def _combine_subgroups(self, sg_id1, sg_id2):
    
        sg_id1 = self._process_subgroup_id(sg_id1)
        sg1, inclusion, restriction = self.subgroup(sg_id1)
        sg_id2 = sg1._process_subgroup_id(sg_id2)
        
        return sg_id2

    def _restrict_irrep(self, irrep: Tuple, id: int) -> Tuple[np.matrix, List[Tuple]]:
        r"""
        
        Restrict the input irrep to the subgroup :math:`C_m` with order ``m``.
        If the current group is :math:`C_n`, it restricts to the subgroup generated by :math:`r^{(n/m)}`.
        Notice that :math:`m` has to divide the order :math:`n` of the current group.
        
        The method takes as input the integer :math:`m` identifying of the subgroup to build (the order of the subgroup)

        Args:
            irrep (tuple): the identifier of the irrep to restrict
            id (int): the integer ``m`` identifying the subgroup

        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """
    
        irr = self.irrep(*irrep)
    
        # Build the subgroup
        sg, _, _ = self.subgroup(id)
    
        order = id
    
        change_of_basis = None
        irreps = []
    
        f = irr.attributes["frequency"] % order
    
        if f > order/2:
            f = order - f
            change_of_basis = np.array([[1, 0], [0, -1]])
        else:
            change_of_basis = np.eye(irr.size)
    
        r = (f,)
    
        irreps.append(r)
        if sg.irrep(*r).size < irr.size:
            irreps.append(r)
        
        return change_of_basis, irreps

    def _build_representations(self):
        r"""
        Build the irreps and the regular representation for this group
        
        """
        
        N = self.order()

        # Build all the Irreducible Representations
        for k in range(0, int(N // 2) + 1):
            self.irrep(k)
            
        # Build all Representations

        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name : irr for irr in self.irreps()})

        # build the regular representation
        self.representations['regular'] = self.regular_representation
        self.representations['regular'].supported_nonlinearities.add('vectorfield')
        
    def _build_quotient_representations(self):
        r"""
        Build all the quotient representations for this group

        """
        for n in range(2, int(math.ceil(math.sqrt(self.order())))):
            if self.order() % n == 0:
                self.quotient_representation(n)
    
    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0']

    def irrep(self, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep of frequency ``k`` of the current cyclic group.
        The frequency has to be a non-negative integer in :math:`\{0, \dots, \left \lfloor N/2 \right \rfloor \}`,
        where :math:`N` is the order of the group.
        
        Args:
            k (int): the frequency of the representation

        Returns:
            the corresponding irrep

        """
        id = (k,)
        
        if id not in self._irreps:
            
            assert 0 <= k <= self.order() // 2, (k, self.order())
            name = f"irrep_{k}"
            
            n = self.order()
            
            if k == 0:
                # Trivial representation
            
                irrep = _build_irrep_cn(0)
                character = _build_char_cn(0)
                supported_nonlinearities = ['pointwise', 'gate', 'norm', 'gated', 'concatenated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                            supported_nonlinearities=supported_nonlinearities,
                                                            character=character,
                                                            # trivial=True,
                                                            frequency=k)
            elif n % 2 == 0 and k == int(n/2):
                # 1 dimensional Irreducible representation (only for even order groups)
                irrep = _build_irrep_cn(k)
                character = _build_char_cn(k)
                supported_nonlinearities = ['norm', 'gated', 'concatenated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 1, 'R',
                                                            supported_nonlinearities=supported_nonlinearities,
                                                            character=character,
                                                            frequency=k)
            else:
                # 2 dimensional Irreducible Representations

                irrep = _build_irrep_cn(k)
                character = _build_char_cn(k)

                supported_nonlinearities = ['norm', 'gated']
                self._irreps[id] = IrreducibleRepresentation(self, id, name, irrep, 2, 'C',
                                                            supported_nonlinearities=supported_nonlinearities,
                                                            character=character,
                                                            frequency=k)
        return self._irreps[id]

    def bl_irreps(self, L: int) -> List[Tuple]:
        r"""
        Returns a list containing the id of all irreps of frequency smaller or equal to ``L``.
        This method is useful to easily specify the irreps to be used to instantiate certain objects, e.g. the
        Fourier based non-linearity :class:`~escnn.nn.FourierPointwise`.
        """
        assert 0 <= L <= self.order() // 2, (L, self.order())
        return [(l,) for l in range(L+1)]

    def _clebsh_gordan_coeff(self, m, n, j) -> np.ndarray:
        m, = self.get_irrep_id(m)
        n, = self.get_irrep_id(n)
        j, = self.get_irrep_id(j)

        rho_m = self.irrep(m)
        rho_n = self.irrep(n)
        rho_j = self.irrep(j)

        if m == 0 or n == 0:
            if j == m + n:
                return np.eye(rho_j.size).reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            else:
                return np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))
        elif (self.N % 2 == 0) and (m == self.N//2 or n == self.N//2):
            if j == m + n:
                return np.eye(rho_j.size).reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            elif j == (self.N -m - n):
                cg = np.eye(rho_j.size)
                if rho_j.size > 1:
                    cg[:, 1] *= -1
                return cg.reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            else:
                return np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))
        else:
            cg = np.array([
                [1., 0., 1., 0.],
                [0., 1., 0., 1.],
                [0., -1., 0., 1.],
                [1., 0., -1., 0.],
            ]) / np.sqrt(2)
            if j == m + n:
                cg = cg[:, 2:]
            elif j == self.N - m - n:
                cg = cg[:, 2:]
                cg[:, 1] *= -1
            elif j == m - n:
                cg = cg[:, :2]
            elif j == n - m:
                cg = cg[:, :2]
                cg[:, 1] *= -1
            else:
                cg = np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))

            return cg.reshape(rho_n.size, rho_m.size, -1, rho_j.size).transpose(1, 0, 2, 3)

    def _tensor_product_irreps(self, J: int, l: int) -> List[Tuple[Tuple, int]]:
        J, = self.get_irrep_id(J)
        l, = self.get_irrep_id(l)
    
        if J == 0 or l == 0:
            return [
                ((l + J,), 1)
            ]
        elif (self.N % 2 == 0) and (J == self.N // 2 or l == self.N // 2):
            j = (J + l) if (J+l <= self.N//2) else (self.N - J - l)
            return [
                ((j,), 1)
            ]
        elif l == J:
            j = (J + l) if (J+l <= self.N//2) else (self.N - J - l)
            m = 1 if j < self.N/2 else 2
            return [
                ((0,), 2),
                ((j,), m),
            ]
        else:
            j = (J + l) if (J+l <= self.N//2) else (self.N - J - l)
            m = 1 if j < self.N/2 else 2
            return [
                ((np.abs(l - J),), 1),
                ((j,), m),
            ]

    _cached_group_instances = {}
    
    @classmethod
    def _generator(cls, N: int) -> 'CyclicGroup':
        if N not in cls._cached_group_instances:
            cls._cached_group_instances[N] = CyclicGroup(N)
        
        return cls._cached_group_instances[N]


def _build_irrep_cn(k: int):
    def irrep(element: GroupElement, k:int =k) -> np.ndarray:
        if k == 0:
            return np.eye(1)
        
        n = element.group.order()

        if n % 2 == 0 and k == int(n / 2):
            # 1 dimensional Irreducible representation (only for even order groups)
            return np.array([[np.cos(k * element.to('radians'))]])
        else:
            # 2 dimensional Irreducible Representations
            return utils.psi(element.to('radians'), k=k)
        
    return irrep


def _build_char_cn(k: int):
    
    def character(element: GroupElement, k=k) -> float:
        if k == 0:
            return 1.
        
        n = element.group.order()
        
        if n % 2 == 0 and k == int(n / 2):
            # 1 dimensional Irreducible representation (only for even order groups)
            return np.cos(k * element.to('radians'))
        else:
            # 2 dimensional Irreducible Representations
            return 2*np.cos(k * element.to('radians'))
    
    return character


def _build_parent_map(G: CyclicGroup, order: int):
    def parent_mapping(e: GroupElement, G: Group = G, order=order) -> GroupElement:
        return G.element(e.to('int') * G.order() // order)
    
    return parent_mapping


def _build_child_map(G: CyclicGroup, sg: CyclicGroup):
    assert G.order() % sg.order() == 0
    
    def child_mapping(e: GroupElement, G=G, sg: Group = sg) -> GroupElement:
        assert e.group == G
        i = e.to('int')
        ratio = G.order() // sg.order()
        if i % ratio != 0:
            return None
        else:
            return sg.element(i // ratio)
    
    return child_mapping


