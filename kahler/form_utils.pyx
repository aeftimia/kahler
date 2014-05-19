__all__ = ['wedge', 'derham_map', 'parity', 'naive_derham_map']

from numpy import arange, asarray, tensordot, zeros, linspace
from itertools import product, combinations, permutations, groupby
from .parallel import parmapreduce, parmap
from functools import reduce

from cython cimport boundscheck, wraparound, profile

def shuffle(S, p):
    shuffles = []
    s = set(S)
    for p_shuffle in combinations(S, p):
        shuffles.append(p_shuffle + tuple(sorted(s - set(p_shuffle))))
    return shuffles    

def parity(S, perm):

    perm = [dict(zip(S,range(len(S))))[x] for x in perm]
    n = len(perm)

    # Decompose into disjoint cycles. We only need to
    # count the number of cycles to determine the parity
    num_cycles = 0
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        num_cycles += 1
        j = i
        while True:
            assert j not in seen
            seen.add(j)
            j = perm[j]
            if j == i:
                break

    return (n - num_cycles) % 2

def wedge(tensor1, tensor2):
    resulting_dimension = tensor1.ndim + tensor2.ndim
    shape = tensor1.shape[0]
    p = tensor1.ndim
    result = zeros((shape,) * resulting_dimension, dtype='complex')
    for combo in combinations(range(shape), resulting_dimension):
        for perm in shuffle(combo, p):
            result[combo] += (-1) ** parity(combo, perm) * tensor1[perm[:p]] * tensor2[perm[p:]]
            
    for combo in combinations(range(shape), resulting_dimension):
        r0 = result[combo]
        for perm in permutations(combo):
            result[perm] = (-1) ** parity(combo, perm) * r0
    return result

def derham_map(fun, points, subdivisions=1):
    dim = points.shape[1] - 1

    if not dim:
        return asarray(parmap(fun, points[:, 0]))

    closure = [x for x in product(*[linspace(0, 1, subdivisions)] * dim) if sum(x) <= 1]

    point_set = []
    factors = []
    integration_factor = 0

    for k, b in groupby(closure, lambda bary: list(bary).count(0) + sum(bary) == 1):
        b = asarray(list(b))
        point_set.append(b)
        f = 0.5 ** k
        factors.append(f)
        integration_factor += b.shape[0] * f

    coordinates = points.swapaxes(0, 1)
    ref = coordinates[0]
    c2 = coordinates[1:] - ref

    integral = 0
    for f, b in zip(factors, point_set):
        bary = tensordot(b, c2, 1)
        integral += asarray(parmapreduce(lambda x: asarray(list(map(fun, x + ref))), bary)) * f

    return integral / integration_factor

def naive_derham_map(fun, points, subdivisions=1):
    dim = points.shape[1] - 1

    if not dim:
        return (1, asarray(parmap(fun, points[:, 0])))

    interior = asarray([x for x in product(*[linspace(0, 1, subdivisions, endpoint=False)[1:]] * dim) if sum(x) < 1])

    if not len(interior):
        return (0, 0)

    coordinates = points.swapaxes(0, 1)
    ref = coordinates[0]
    c2 = coordinates[1:] - ref

    bary = tensordot(interior, c2, 1)

    return (len(interior), asarray(parmapreduce(lambda x: asarray(list(map(fun, x + ref))), bary)))