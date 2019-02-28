"""
A collection of useful functions for problems in quantum information theory.

"""


import numpy as np
from operators import op1
from itertools import product
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from matplotlib import cm, rcParams

# compute effect of a CPTP map


def CPTP(kraus, rho):

    nrho = np.zeros(rho.shape, dtype='complex128')
    for i in kraus:
        nrho += np.dot(i, np.dot(rho, i.conj()))
    return nrho


# generates tensor products of arrayset according to combination
def kronjob(arrayset, combination):
    # initialise array
    matrix = 1
    # construct appropriate tensor product
    for el in combination:
        matrix = np.kron(matrix, arrayset[int(el)])
    return matrix


def dagger(M):
    """
    Return conjugate transpose of input array
    """
    return np.transpose(np.conjugate(M))


def eye_like(m, dtype=np.complex128):
    """
    Returns identity matrix with same dims as square matrix m.
    """
    return np.eye(np.shape(m)[0], dtype=dtype)

# computes the partial trace of density operator m \in L(H_d), tracing out subsystems in the list sys
def partialtrace(m, sys):
    # type enforcement
    m = np.asarray(m)
    # sort subsystems
    sys = sorted(sys)
    # get tensor dimensions
    qnum = int(round(np.log2(len(m))))
    # compute dimensions of tensor
    tshape = (2,)*2*qnum
    # reshape to tensor
    mtensor = m.reshape((tshape))
    # compute dimensions to trace over
    index1, index2 = sys[0], sys[0] + qnum
    del sys[0]
    newdim = 2**(qnum-1)
    # compute reduced density matrix via recursion
    if len(sys) > 0:
        # trace out target subsystem (repeated reshaping is a bit rough but its not worth the fix time)
        mtensor = np.trace(mtensor, axis1=index1,
                           axis2=index2).reshape((newdim, newdim))
        # adjust subsequent target dimensions with shallow copy
        sys[:] = [i-1 for i in sys]
        # by the power of recursion!
        mtensor = partialtrace(mtensor, sys)
    else:
        # bottom of the pile, compute and pass up the chain
        mtensor = np.trace(mtensor, axis1=index1,
                           axis2=index2).reshape((newdim, newdim))
    return mtensor

# generates an operator spanning set for any system with qnum qubits


def opbasis(qnum):
    # define operator basis set (Pauli set in this case)
    opset = np.asarray([op1['id'],
                        op1['x'],
                        op1['y'],
                        op1['z']])

    # determine all combinations
    combs = iter(''.join(seq)
                 for seq in product(['0', '1', '2', '3'], repeat=qnum))
    operbasis = []
    # construct density basis
    for item in combs:
        operbasis.append(kronjob(opset, item))
    operbasis = np.asarray(operbasis)

    # return operator basis
    return operbasis

def bell_gen(N=2):
    """"
    Generate an N qubit Bell state |psi^+>
    """

    bell = np.zeros((2**N,2**N), dtype=np.complex128)

    # exploit structure rather than constructing a circuit
    bell[0,0] = 0.5
    bell[-1,0] = 0.5
    bell[0,-1] = 0.5
    bell[-1,-1] = 0.5

    return bell

def rho_gen(N=1):
    """
    Generates a spanning set of density matrices for N qubits
    """
    dim = 2**N

    # initialise basis set
    rhob = np.empty((dim**2, dim, dim), dtype=np.complex128)
    # define component set
    rhobase = np.empty((4, 2, 2), dtype=np.complex128)
    rhobase[0, :, :] = np.asarray([[1, 0], [0, 0]])
    rhobase[1, :, :] = np.asarray([[0, 0], [0, 1]])
    rhobase[2, :, :] = np.asarray([[1, 1], [1, 1]])/2
    rhobase[3, :, :] = np.asarray([[1, -1j], [1j, 1]])/2

    # generate permutation list
    combs = product(['0', '1', '2', '3'], repeat=N)
    for i, comb in enumerate(combs):
        rho = 1.0
        for j in comb:
            rho = np.kron(rho, rhobase[int(j), :, :])

        rhob[i, :, :] = rho
    return rhob


def dual_gen(rhob, N=1):
    """
    Computes the duals of an input basis <rhob>.
    """
    # get system dimension
    dim = np.shape(rhob)[-1]

    # define pauli basis
    pauli = opbasis(N)
    # reshape pauli array
    basis_flat = np.transpose(np.reshape(pauli, [dim**2]*2, order='C'))
    # initialise coefficent array
    coeffs = np.empty((2**(2*N), 2**(2*N)), dtype=float)

    # compute basis coefficients (will need to reformulate with pyconv + constraints)
    for i in range(int(dim**2)):
        rho = np.reshape(rhob[i, :, :], [dim**2, 1])

        # could compute analytically but I want this to be generalisable
        coeffs[i, :] = np.real(np.squeeze(np.linalg.solve(basis_flat, rho)))

        # check that reconstructed matrices are within tolerance
        rhor = np.zeros(([dim, dim]), dtype=np.complex128)
        for j in range(dim**2):
            rhor += coeffs[i, j] * np.reshape(basis_flat[:, j], [dim, dim])
        assert np.allclose(rhor, np.reshape(rho, [
                           dim, dim]), atol=1e-9), "Reconstructed array not within tolerance of target: Aborting"

    # find the inverse of the coefficient matrix
    F = np.conjugate(np.transpose((np.linalg.inv(coeffs))))

    # compute the duals to rhob
    duals = np.zeros_like(rhob)
    for i in range(dim**2):
        for j in range(dim**2):
            duals[i, :, :] += 0.5*F[i, j] * \
                np.reshape(basis_flat[:, j], [dim, dim])
    return duals


def dualp_tomography(dual, rhop):
    """
    Perform process tomography using the dual approach and outputs the Choi state of the process
    Inputs - set of duals to the input states and the output states.
    dual = ndarray(d^2, d, d)
    outputs = ndarray(d^2, d, d)
    """
    # get dimensionality of system
    dim = np.shape(rhop)[-1]

    # initialise Aform and Bform
    Aform = np.zeros((dim**2, dim**2), dtype=np.complex128)
    Bform = np.zeros((dim**2, dim**2), dtype=np.complex128)

    # compute A form in terms of the duals
    for i in range(dim**2):
        Aform += np.outer(rhop[i, :, :], np.conjugate(dual[i, :, :]))

    # compute B form in terms of the duals
    for j in range(dim**2):
        Bform += np.kron(rhop[j, :, :], np.conjugate(dual[j, :, :]))

    # ought to add a check that index transformation of A<->B gives the same as above
    return Aform, Bform


def IA_gen(N):
    """
    Computes the A/B--form of the identity channel
    """
    # define a basis set (and hence the output state)
    rhop = rho_gen(N=N)
    # define duals
    duals = dual_gen(rhop, N=N)
    A, B = dualp_tomography(duals, rhop)
    return A, B


def UA_gen(U):
    """
    Performs process tomography on an input unitary U
    """
    # get dimension of operator space
    N = int(np.round(np.log2(len(U))))
    # generate spanning set
    rhob = rho_gen(N=N)
    # compute duals
    duals = dual_gen(rhob, N=N)
    # compute process effect on spanning set
    rhop = np.copy(rhob)

    # evolve spanning set under unitary
    for i, rho in enumerate(rhob):
        rhop[i, :, :] = U @ rho @ np.conjugate(np.transpose(U))

    # compute A and B form of map
    A, B = dualp_tomography(duals, rhop)
    return A, B


def AB_join(A1, A2):
    """
    Joins two A (or B? - investigate) form maps into a single operator on the joint Hilbert space. 
    Remap requirements comes from ordering mismatch between state tensor product and A-form tensor product.
    """
    # compute tensor product and get output dimension
    joint = np.kron(A1, A2)
    dim = len(joint)

    # explicitly compute sub--process dimensions
    A1_dim = len(A1)
    A2_dim = len(A2)

    # local system dimensions
    A1_sdim = int(round(np.sqrt(A1_dim)))
    A2_sdim = int(round(np.sqrt(A2_dim)))

    # construct subsystem remap
    cshape = [A1_sdim, A1_sdim, A2_sdim, A2_sdim]*2

    return np.reshape(np.transpose(np.reshape(joint, cshape), [0, 2, 1, 3, 4, 6, 5, 7]), [dim, dim])


def tracenorm(m):
    import numpy.linalg
    return np.sum(np.abs(numpy.linalg.eigh(m)[0]))


def AB_shuffle(form):
    """
    Switches between the A/B form of a map. Assumes same input/output dimensions.
    """
    # get dimension of map
    dim = len(form)
    # get subsystem dimension
    sub_dim = int(round(np.sqrt(dim)))
    # reshape subsystems into 4-tensor
    return np.reshape(np.transpose(np.reshape(form, [sub_dim]*4), (0, 2, 1, 3)), (dim, dim))


def random_U(N, num=1):
    """
    Generate num random unitaries on N qubits using simple method
    """
    # preallocate U array
    Us = np.zeros((2**N,2**N,num), dtype=np.complex128)

    # generate unitaries using naive method
    for i in range(0,num):
        # generate a random complex matrix (yes I know I could do this in one go rather than iterate)
        U = np.random.rand(2**N,2**N) + 1j*np.random.rand(2**N,2**N) 

        # QR factorisation
        [Q,R] = np.linalg.qr(U/np.sqrt(2))

        R = np.diag(np.diag(R)/np.abs(np.diag(R)))
        # compute the unitary
        Us[:,:,i] = Q @ R

    return Us


def haar_sample(N=1, num=10):
    """
    Generates <num> quantum states of dimension 2**N from Haar distribution.
    """
    # get dimension of system
    dim = int(2**N)
    # generate random complex arrays
    states = np.random.uniform(low=-1, high=1, size=(num, dim, dim)) + \
        np.random.uniform(low=-1, high=1, size=(num, dim, dim))*1j
    for i in range(num):
        # compute Hilbert Schmidt norm
        A2 = np.sqrt(np.trace(dagger(states[i, :, :]) @ states[i, :, :]))
        # normalise
        states[i, :, :] /= A2
        # compute random density matrix
        states[i, :, :] = dagger(states[i, :, :]) @ states[i, :, :]

    return states


def rhoplot(rho, axislabels=None, save=False):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.mplot3d import Axes3D

    # extract real and imaginary components of density matrix
    realrho = np.real(rho)
    imagrho = np.imag(rho)

    # instantiate new figure
    fig = plt.gcf()
    fig.canvas.set_window_title('Density Plot')
    #rax = Axes3D(fig)
    rax = fig.add_subplot(121, projection='3d')
    iax = fig.add_subplot(122, projection='3d')

    # set titles
    rax.title.set_text('Real$(\\rho)$')
    iax.title.set_text('Imag$(\\rho)$')
    # apply custom labelling
    if axislabels is not None:
        rax.set_xticklabels(axislabels)
        rax.set_yticklabels(axislabels)
        iax.set_xticklabels(axislabels)
        iax.set_yticklabels(axislabels)

    # dimension of space
    dim = np.shape(realrho)[0]
    # create indexing vectors
    x, y = np.meshgrid(range(0, dim), range(0, dim), indexing='ij')
    x = x.flatten('F')
    y = y.flatten('F')
    z = np.zeros_like(x)

    # create bar widths
    dx = 0.5*np.ones_like(z)
    dy = dx.copy()
    dzr = realrho.flatten()
    dzi = imagrho.flatten()

    # compute colour matrix for real matrix and set axes bounds
    norm = colors.Normalize(dzr.min(), dzr.max())
    rcolours = cm.BuGn(norm(dzr))
    rax.set_zlim3d([0, np.max(dzr)])
    iax.set_zlim3d([0, np.max(dzr)])

    inorm = colors.Normalize(dzi.min(), dzi.max())
    icolours = cm.jet(inorm(dzi))

    # plot image
    rax.bar3d(x, y, z, dx, dy, dzr, color=rcolours)
    iax.bar3d(x, y, z, dx, dy, dzi, color=icolours)
    #plt.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    plt.show()


def vecstate(state):
    """
    Vectorises state using the computational basis or devectorises.
    """
    # get dimension of first axis
    dim = np.shape(state)[0]

    if dim == np.shape(state)[1]:
        return np.reshape(state, [dim**2, 1])
    else:
        sdim = int(round(np.sqrt(dim)))
        return np.reshape(state, [sdim, sdim])


def subsyspermute(rho, perm, dims):
    # get dimensions of system
    d = np.shape(rho)
    # get number of subsystems
    sys = len(dims)
    # perform permutation
    perm = [(sys - 1 - i) for i in perm[-1::-1]]
    perm = listflatten([perm, [sys + j for j in perm]])
    return np.transpose(rho.reshape(dims[-1::-1]*2), perm).reshape(d)


def qre(rho, gamma):
    """
    computes the quantum relative entropy between two states rho and gamma
    """
    return np.trace(np.dot(rho, (logm(rho) - logm(gamma))))


def kolmogorov(rho, gamma):
    """
    Computes the trace or Kolmogorov distance between two quantum states
    """
    return np.trace(abs(rho-gamma))/2


def qfid(rho, gamma):
    """
    Computes the quantum fidelity between two quantum states (not a metric)
    """
    print(sqrtm(rho))
    return (np.trace(sqrtm(sqrtm(rho)*gamma*sqrtm(rho))))**2


def bures(rho,gamma):
    """
    Computes Bures angle between two states
    """
    return np.arccos(np.clip(np.sqrt(np.trace(sqrtm(sqrtm(rho) @ gamma @ sqrtm(rho)))**2), 0.0,1.0))


def helstrom(rho, gamma):
    """
    Computes the Helstrom distance between two quantum states
    """
    return sqrtm(2*(1-sqrtm(qfid(rho, gamma))))


def isdensity(rho):
    """
    Checks if an input matrix is a valid density operator
    """
    res = True
    res &= np.all(np.isclose(rho - dagger(rho), 0.0))  # symmetric
    res &= np.all(np.linalg.eigvals(rho) >= 0)         # positive semidefinite
    res &= np.isclose(np.trace(rho), 1.0)              # trace one
    return res


def mem_check(dims, type=np.float64):
    """
    Returns the number of bits an array with dimensions <dims> and datatype <type> will require.
    """
    return np.prod(dims)*np.finfo(type).bits


def povm_gen(N=1):
    """
    Generates a simple POVM for N qubits. Corresponds to a spanning set of the NxN Hermitian matrices.
    """
    # explicitly calculate for later modifcation to generalised subsystems
    dim = 2**N
    # initialise set array
    povm = np.empty((dim**2, dim, dim), dtype=np.complex128)
    # Set of N matrices with one 1 on the diagonal union with N(N−1)/2 [1,+i/-i] on off--diagonal.
    # define N=1 spanning set and build from there
    povmbase = np.empty((4, 2, 2), dtype=np.complex128)
    alpha = (np.sqrt(3)*1j-1)/2
    alphac = np.conjugate(alpha)
    povmbase[0, :, :] = np.asarray([[1, 0], [0, 0]])/2
    povmbase[1, :, :] = np.asarray([[1, np.sqrt(2)], [np.sqrt(2), 2]])/6
    povmbase[2, :, :] = np.asarray(
        [[1, np.sqrt(2)*alpha], [np.sqrt(2)*alphac, 2]])/6
    povmbase[3, :, :] = np.asarray(
        [[1, np.sqrt(2)*alphac], [np.sqrt(2)*alpha, 2]])/6

    # since larger Hilbert spaces are simply chained tensor products, so too will the joint basis set 
    # be the tensor product of the constituent spaces.
    # generate permutation list
    combs = product(['0', '1', '2', '3'], repeat=N)
    for i, comb in enumerate(combs):
        pvm = 1.0
        for j in comb:
            pvm = np.kron(pvm, povmbase[int(j), :, :])

        povm[i, :, :] = pvm
    return povm




class ControlSpan():
    """
    Computes a spanning set of control operations (the space B(B(H_d)) with d=2**N). Assumes qubits for the moment.
    Must be done as an iterator else the memory requirements are simply too severe for even short processes. 
    """
    def __init__(self, N=1, k=1):

        # number of time steps in process
        self.k = k
        # number of qubits controls will be applied too
        self.N = N
        # compute spanning set of density matrices
        self.rhob = rho_gen(N=N)
        # compute POVM
        self.povm = povm_gen(N=N)


    def __iter__(self):
        """
        initialise control sequence number and loop constants
        """ 
        self.cseq_num = 0
        # preallocate current control sequence
        self.control_sequence = np.empty((self.k, 2**(2*self.N),2**(2*self.N)), dtype=np.complex128)
        # current rho to iterate over
        self.rho_num = [0]*self.k
        # current povm to iterate over
        self.povm_num = [0]*self.k

        return self


    def __next__(self):
        """
        Iteratively generate the control maps for a k--step process tensor
        """
        
        # compute control sequence
        if self.rho_num[-1] < len(self.rhob):

            # catch start sequence case
            if self.cseq_num == 0:
                for i in range(0, self.k):
                    rho_sel = self.rho_num[i]
                    povm_sel = self.povm_num[i]
                    self.control_sequence[i, :,:] = np.kron(self.rhob[rho_sel,:,:], self.povm[povm_sel,:,:])
                    self.cseq_num += 1
                    return self.control_sequence

            # perform incrementation of counters
            inc_flag = True
            inc_ind = 0
            while inc_flag:
                # check if incrementation overflows
                if self.povm_num[inc_ind]+1>=len(self.povm):
                    inc_ind += 1

                    # check if rhob needs to be incremented
                    if inc_ind >= len(self.povm_num):
                        inc_ind = 0
                        while inc_flag:
                            if self.rho_num[inc_ind]+1 >= len(self.rhob):
                                inc_ind += 1
                                # exit if we have are at the end of the final iter
                                if inc_ind == len(self.rho_num):
                                    raise StopIteration

                            else:
                                self.rho_num[inc_ind] += 1
                                self.rho_num[:inc_ind] = [0]*inc_ind
                                self.povm_num = [0]*len(self.povm_num)
                                inc_flag = False

                else:
                    self.povm_num[inc_ind] += 1
                    self.povm_num[:inc_ind] = [0]*inc_ind
                    inc_flag = False

            # iterate over number of time steps
            for i in range(0, self.k):
                rho_sel = self.rho_num[i]
                povm_sel = self.povm_num[i]
                self.control_sequence[i, :,:] = np.kron(self.rhob[rho_sel,:,:], self.povm[povm_sel,:,:])

            # iterate loop couinter
            self.cseq_num += 1
            return self.control_sequence

        else:
            raise StopIteration


def Universal_U():
    """
    Generates an interesting unitary drawing from the universal gate set
    """

    # base unitary
    U = np.eye(2**5)

    # hadamard combo to get started
    H = np.kron(kronjob([op1['h']],[0,0]), np.eye(8))
    U = H @ U

    # controlled nots
    U = kronjob([op1['cx'], op1['id']], [0,0,1]) @ U

    # some local operators
    U = kronjob([op1['id'],op1['t'],op1['s']],[2,1,2,0,0]) @ U

    # some more controlled nots
    U = kronjob([op1['cx'], op1['id']], [1,0,1,1]) @ U

    # hit some phase gates
    U = kronjob([op1['s'], op1['id']], [1,1,0,1,1]) @ U

    # finally some T gate action
    U = kronjob([op1['id'],op1['t'],op1['s']],[1,1,1,2,0])

    return U


class ProcessTensor():
    """
    Class that defines the process tensor given specified simulation parameters. Assumes qubits for now. 
    """

    def __init__(self, rho_se, U, sd=2, k=1, force=False):
        # store input variables
        # the initial system environmental state
        self.rho_se = rho_se
        # Unitary operator that is applied on the system/environment between controls
        self.U = U
        # the Hilbert space dimension of the system
        self.sd = sd
        # the number of timesteps to simulate for
        self.k = k
        # whether to force simulation for questionable inputs
        self.force = force

        # compute utility parameters for later use
        # number of qubits in system and environment
        self.sq = int(np.round(np.log2(self.sd)))
        self.eq = int(np.round(np.log2(len(self.rho_se)))) - self.sq
        # dimension of environmental subsystem (assumes qubits)
        self.ed = 2**self.eq

        # check input parameters are valid
        assert np.shape(self.U)[0] == np.shape(self.U)[1] and len(np.shape(
            U)) == 2, "Unitary must be a square matrix but has dimensions: {}".format(np.shape(U))
        assert isdensity(
            self.rho_se), "Initial system/environment must be a valid density operator"
        assert np.shape(self.U)[1] == np.shape(self.rho_se)[
            0], "Unitary and initial state dimension mismatch: {} and {}".format(np.shape(self.U, self.rho_se))

    def apply(self, A, env=True):
        """
        Apply a sequence of control operations, assumes A is a DxDxk matrix made up of k A--forms. If env is false 
        will return just the system subsystem state i.e. will trace out the environment
        """

        # check if process tensor will be too large (not needed for now)
        if self.k > 5 and not self.force:
            print("a {} step process is very large, set force parameter to True to compute this process tenor".format(self.k))

        # assert that the length of the controls is less than the time length of the process tensor
        try:
            assert np.shape(A)[2] == self.k, "Number of control operations does not equal k length of process tensor: {} != {}".format(
                self.k, np.shape(A)[2])

        except AssertionError as e:
            # catch force case
            if self.force:
                pass
            else:
                raise AssertionError(e)

        # simple evaluation of process tensor
        # create copy of inital system/environment to evolve and vectorise
        rho = vecstate(np.copy(self.rho_se))
        # compute identity channel to apply to environment system on control step (assumes qubits)
        env_identity, _ = IA_gen(N=self.eq)

        # iterate over time steps, performing control then unitary
        for step in range(0, self.k):
            # extract control operation to perform - assume A--form
            A_step = A[:, :, step]
            # pad with identity channel acting on environmental subsystem
            control_op = AB_join(env_identity, A_step)
            # apply control operation channel
            rho = control_op @ rho
            # convert to density operator and apply unitary (avoids calculating the channel rep of the unitary)
            rho = vecstate(self.U @ vecstate(rho) @ dagger(self.U))

        # devectorise final output state
        rho = vecstate(rho)

        # return full system state by default or trace out environment if requested
        if env:
            return np.asarray(rho)
        else:
            # list of qubits in environmental system to trace out
            t_qubits = list(range(0, self.eq))
            return partialtrace(rho, t_qubits)

    def pt_tomog(self):
        """
        Perform process tomography on the process tensor. Easiest way of discerning the full map if a bit
        computationally intensive. Can be done more efficiently with direct calculation but not by much and
        it is far easier to make a mistake.
        """

        # ensure the memory requirements are not beyond us (limiting case is storing the duals)
        if not self.force:
            # abort if required memory is more than 2 GB
            if mem_check([self.k], type=np.complex) > 1.6e+10:
                raise MemoryError(
                    'Process tensor dimension is too large (set force parameter to override)')

        # construct control operation iterator
        controls = ControlSpan(N=self.sq, k=self.k)

        # preallocate process tensor array
        ptensor = np.zeros((),dtype=np.complex128)

        

    def pt_compute(self): 
        """
        Directly compute the process tensor, sidestepping any tomography calculations. A very unpleasant function 
        to write, due solely to the subsystem operations that need to be performed. 
        """

    def cji(self):
        """
        Compute the Choi-Jamiokowlski form of the process tensor. We can't save anywhere on memory requirements 
        so we may as welll minimise the amount of time it takes. 
        """
        pass
