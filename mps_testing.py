import mpnum as mp
import numpy as np
from operators import *

CX = np.array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.],
               [ 0.,  0.,  1.,  0.]])

CX_arr = CX.reshape([2,2,2,2])

CX_mpo = mp.MPArray.from_array_global(CX_arr, ndims=2)


vec = np.kron([0,1], [0,1])

vec_arr = vec.reshape([2,2])

mps = mp.MPArray.from_array(np.array([1,0]), ndims=1)

proj = mpo_dict['y'] #mp.mparray.chain([mpo_dict['x'], mpo_dict['z']])

#out = mp.dot(proj, mps)

#print(len(out), out.ndims, out.ranks)

def info(m):
    """
    prints relevant MPA information.
    """
    print(len(m), m.ndims, m.ranks)


# convert state to MPO
mpo = mp.mpsmpo.mps_to_mpo(mps)

print(mp.trace(mp.dot(proj,mp.mpsmpo.mps_to_mpo(mps))))
