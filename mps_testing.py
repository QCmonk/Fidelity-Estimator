import mpnum as mp
import numpy as np

CX = np.array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.],
               [ 0.,  0.,  1.,  0.]])

CX_arr = CX.reshape([2,2,2,2])

CX_mpo = mp.MPArray.from_array_global(CX_arr, ndims=2)

print(CX_mpo.ndim)

vec = np.kron([0,1], [0,1])

vec_arr = vec.reshape([2,2])

mps = mp.MPArray.from_array(vec_arr, ndims=1)

out = mp.dot(CX_mpo, mps)
print(out.to_array().ravel())

