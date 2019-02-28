
import mpnum as mp
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# Plancks constant
pbar = 6.626070040e-34
# reduced
hbar = pbar/(2*np.pi)
# Bohr magneton in J/Gauss
mub = (9.274009994e-24)/1e4
# g factor
gm = 2.00231930436
# Gyromagnetic ratio
gyro = 699.9e3

# identity matrix
_ID = np.array([[1, 0], [0, 1]])
# X gate
_X = np.array([[0, 1], [1, 0]])
# Z gate
_Z = np.array([[1, 0], [0, -1]])
# Hadamard gate
_H = (1/np.sqrt(2))*np.array([[1, 1], [1, -1]])
# Y Gate
_Y = np.array([[0, -1j], [1j, 0]])
# S gate
_S = np.array([[1, 0], [0, 1j]])
# Sdg gate
_Sdg = np.array([[1, 0], [0, -1j]])
# T gate
_T = np.array([[1, 0], [0, (1 + 1j)/np.sqrt(2)]])
# Tdg gate
_Tdg = np.array([[1, 0], [0, (1 - 1j)/np.sqrt(2)]])
# R gate
_R = np.array([[0,1-1j],[1+1j,0]])/np.sqrt(2)
# CNOT gate
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# CNOT inverse
_CXdg = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
# SWAP gate
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]])
# toffoli gate
_TOFFOLI = block_diag(_ID, _ID, _CX)
# zero state
_pz = np.array([[1,0],[0,0]])
# one state
_po = np.array([[0,0],[0,1]])

def dagger(a):
        return np.transpose(np.conjugate(a))

# define operators for spin 1/2 
op1 = {'h':   _H,
        'id':  _ID,
        'x':   _X,
        'y':   _Y,
        'z':   _Z,
        't':   _T,
        'tdg': _Tdg,
        's':   _S,
        'sdg': _Sdg,
        'r': _R,
        'cx':  _CX,
        'cxdg': _CXdg,
        'swap': _SWAP,
        'toff': _TOFFOLI,
        'pz': _pz,
        'po': _po}

# define matrix product states for useful operators
mpo_dict = {'id': mp.MPArray.from_array_global(op1['id'], ndims=2),
            'h': mp.MPArray.from_array_global(op1['h'], ndims=2),
            'x': mp.MPArray.from_array_global(op1['x'], ndims=2),
            'y': mp.MPArray.from_array_global(op1['y'], ndims=2),
            'z': mp.MPArray.from_array_global(op1['z'], ndims=2),
            's': mp.MPArray.from_array_global(op1['s'], ndims=2),
            't': mp.MPArray.from_array_global(op1['t'], ndims=2),
            'cx': mp.MPArray.from_array_global(op1['cx'].reshape([2,2,2,2]), ndims=2)}




# define operators for spin 1 
op2 = {
		
	}


# measurement projections for spin 1/2
meas1 = {"0":np.asarray([[1,0]]),
		 "1":np.asarray([[0,1]]),
		 "+":np.asarray([[1,1]]/np.sqrt(2)),
		 "-":np.asarray([[1,-1]]/np.sqrt(2)),
		 "+i":np.asarray([[1,1j]]/np.sqrt(2)),
		 "-i":np.asarray([[1,-1j]]/np.sqrt(2)),
		}