import mpnum as mp
import numpy as np
from qinfo import *
from operators import *
from scipy.linalg import expm
from itertools import product, combinations
import matplotlib.pyplot as plt

# TODO: Refactor to allow for subsystems of arbitrary dimension, not just qubits
# TODO: Add better noise generator - not efficiently using MPO structure to generate and so run out of memory for large systems

# set random seed for repeatable simulation
np.random.seed(1234)


def rand_res(probs=[0.5, 0.5], outcomes=[0, 1], num=1):
    """
    Returns a set of <num> <outcomes> using the probability weighting <probs> given.
    """
    # sanitise input probabilities (floating point error can lead to negative/complex values
    probs = np.clip(np.real(probs), 0, 1)
    return np.random.choice(outcomes, size=(1, num), p=probs)[0]


def mle(xi):
    """
    Returns the estimate for the angle
    """
    return np.sum(xi)/len(xi)


def var(theta, esum, n):
    """
    estimates the variance of the parameter
    """
    H = -esum/theta**2 - (n-esum)/((1-theta)**2)
    return (-H)**(-1)


def eye_like(m):
    """
    Returns identity matrix with same dims as square matrix m.
    """
    dim = np.shape(m)[0]
    return np.eye(dim)


def theta_compute(stab_sum, N):
    """
    Compute the angle of rotation for an N qubit estimator.
    """
    modi = 2**(N/2+1)-1-2**(N/2-1)
    if stab_sum < 2**(N/2+1)-1-2**(N/2-1)*2:
        stab_sum = 2**(N/2+1)-1-2**(N/2-1)*2

    return np.arccos((np.round(stab_sum, 5)-modi)/2**(N/2-1))*0.5


def angle_estimate(stabilisers, state, N, shots=500):
    """
    Compute Bures angle given the perturbed state and the stabilisers for the True state
    """

    # initialise total
    stab_sum = 0.0
    # iterate over stabilisers
    for stab in stabilisers:
        # compute out probability given perturbed state
        prob = (1 + mp.trace(mp.dot(stab, state)))/2
        stab_sum += mle(rand_res([1-prob, prob], num=int(shots)))

    theta = theta_compute(stab_sum, N=N)
    # TODO: Fix this uncertainty calculation issue
    uncert = 0  # np.sqrt(var(theta, stab_sum, int(N)))
    return theta, uncert


def bures_mps(rho, sigma):
    """
    Computes the Bures angle for two matrix product states. Requires
    rho and sigma to be pure and input as MPOs
    """
    fid = mp.trace(mp.dot(rho, sigma))

    return np.arccos(np.clip(np.sqrt(fid), 0.0, 1.0))


class random_MPUgen(object):
    """
    Generator producing semi-nonlocal unitaries without end
    """

    def __init__(self, N, max_size=2, relerr=1e-6):
        # store details of Hilbert space
        self.N = N
        self.max_size = min([max_size, N//2])

        # setup iterator
        self.relerr = relerr

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # generate a random unitary and convert to MPO
        mpu = mp.MPArray.from_array_global(
            random_U(self.max_size, num=1).reshape([2]*(self.max_size*2)), ndims=2)

        # number of subsystems we need to appl
        subsys = int(self.N//2) - self.max_size

        # iterate if still have remaining subsystems
        while subsys > 0:
            # generate another unitary
            dim = min([subsys, self.max_size])
            mpo_rand = mp.MPArray.from_array_global(
                random_U(dim, num=1).reshape([2]*(dim*2)), ndims=2)

            # chain
            mpu = mp.chain([mpu, mpo_rand])

            # compress matrix product
            mpu.compress("svd", relerr=relerr)

            # remove computed subsystem dimensions from count
            subsys -= dim

        # chain with identity MPO
        mpu = mp.chain([mpu, mp.chain([mpo_dict["id"]]*(self.N//2))])
        # perform in-place compression on output operator
        mpu.compress("svd", relerr=self.relerr)

        return mpu


def info(m):
    """
    prints relevant MPA information.
    """
    print(len(m), m.ndims, m.ranks)


def mpojob(mpas, selector, relerr=1e-6):
    """
    stacks the set of mpas in the order specified by selector
    """
    prod_chain = []
    for sel in selector:
        prod_chain.append(mpas[sel])

    # compute tensor chain
    chained = mp.chain(iter(prod_chain))
    # compress output rank to within tolerance
    chained.compress("svd", relerr=relerr)

    return chained


def mporep(mpa, mpb, reps, relerr=1e-6):
    """
    constructs the MPA [mpa, mpb_1, mpb_2....mpb_reps]
    """
    # construct MPA iterable
    prod_chain = [mpb]*reps
    prod_chain.insert(0, mpa)

    # construct tensor chain
    chained = mp.chain(iter(prod_chain))
    # compress
    chained.compress("svd", relerr=relerr)

    return chained


def bell_gen(N=2):
    """
    Generates the matrix product state of an N qubit Bell state |psi^+>.
    Nice bit of code. 
    """

    # first generate an N qubit state in |0>^\otimes N
    mps = mp.MPArray.from_kron([np.array([1, 0])]*N)

    # generate entanglement operator in MPO form
    hadamard_mpo = mporep(mpo_dict["h"], mpo_dict["id"], reps=N-1)
    cx_mpo = mp.chain([mpo_dict["id"]]*N)
    for i in range(0, N-1):
        # construct selector for iteration
        selector = [0]*(N-1)
        selector[i] = 1
        # construct CX stage
        cx_mpo_stage = mpojob([mpo_dict["id"], mpo_dict["cx"]], selector)

        # add to cx sequence
        cx_mpo = mp.dot(cx_mpo_stage, cx_mpo)

    # define entangling operation in MPO form
    entangle_op = mp.dot(cx_mpo, hadamard_mpo)
    # compress to minimise memory overhead (TODO: must add overlap check)
    entangle_op.compress("svd", relerr=1e-6)

    # compute Bell state in MPS form
    bell = mp.dot(entangle_op, mps)

    return bell


def bell_stab_gen(N=2):
    """
    Compute the generators for an N qubit Bell state |\\psi^+>.
    """
    # basic input check
    assert N % 2 == 0, "Bell states must have even number of qubits"

    # default XXX...X generator

    generators = [mpojob([mpo_dict['x']], [0]*N)]

    # iterate over maximum number of generators needed (Lemma 2)
    for i in range(1, int(N/2) + 1):
        # placement vector
        selector = list([0]*N)
        # select Z_n otimes Z_n-1
        selector[i-1:i+1] = (1, 1)
        # construct stabiliser generator
        new_gen = mpojob([mpo_dict["id"], mpo_dict["z"]], selector)
        # add to set
        generators.append(new_gen)

    return generators


def operator_find(gen, N, relerr=1e-6):
    """
    Computes all unique stabilisers given the list from gen. Assumes Pauli operators.
    """
    # stabiliser storage
    stabilisers = []

    # max depth of stabiliser
    max_depth = len(gen)+1

    # list of generator identifiers
    indices = range(0, len(gen))

    # iterate over length N stabiliser
    for i in range(1, max_depth):

        # generate every unique generator sequence - assumes Pauli based generators
        gen_combs = set(list(combinations(indices, i)))

        # iterate over index combinations
        for comb in gen_combs:

            # create new
            stab = mpojob([mpo_dict["id"]], [0]*N)
            for k in comb:

                # add generator to stabiliser sequence
                stab = mp.dot(gen[k], stab)

            # add new stabiliser to set
            stabilisers.append(stab)

    return stabilisers


def UniU_error(U, N, perturbations=100, exact=True, shots=8000):
    """
    Check estimation error for an input U, number of perturbations and wether to use exact or finite sampling.
    """

    # generate entangled states
    rho = bell_gen(N=N)

    # generate Bell state stabilisers
    bell_gstab = bell_stab_gen(N=N)

    # convert to MPO if required
    if type(U) != mp.mparray.MPArray:
        try:
            U = mp.MPArray.from_array_global(U.reshape([2]*N*2), ndims=2)
        except ValueError:
            # catch operator dimension mismatch
            raise ValueError("Cannot reshape unitary into MPO, check dimensions")

    # apply to entangled state
    rho = mp.dot(U, rho)

    # evolve generators under unitary
    gstab = [mp.dot(mp.dot(U, stb), U.adj()) for stb in bell_gstab]

    # generate stabiliser set
    stabilisers = operator_find(gstab, N=N)

    # apply to entangled state and convert to MPO for measurement phase
    rho = mp.mpsmpo.mps_to_mpo(rho)

    # calculate the estimation error for requested number of perturbations
    error = []

    # initialise random unitary generator
    U_perturb = random_MPUgen(N)

    for i in range(0, perturbations):
        print("Now computing unitary perturbation {}\r".format(
            i), end="", flush=True)

        # make a copy
        rho_c = rho.copy()

        # compute a local perturbation using generator
        U_p = next(U_perturb)

        # apply to Choi state
        rho_c = mp.dot(mp.dot(U_p, rho_c), U_p.adj())

        # compute expectation values exactly or with finite samples
        if exact:
            Q = 0.0
            
            # iterate over stabiliser measurements
            for stab_proj in stabilisers:    
                # add to Q sum
                Q += (1 + mp.trace(mp.dot(stab_proj, rho_c)))/2
                print(Q)

            # estimate angle
            a_est = theta_compute(Q, N=N)

        else:
            # estimate expectation values from finite number of outcomes
            a_est, a_uncert = angle_estimate(
                stabilisers, rho_c, N=N, shots=shots)

            if np.abs(np.real(bures_mps(rho, rho_c) - a_est)) > 0.5:
                print("High estimation error: {:.3f}, something has gone wrong".format(a_est))
                continue
   
        # compute angle estimate error
        error.append(np.real(bures_mps(rho, rho_c) - a_est))

    if exact:
        # output average estimation error - should always be small (<1e-4 depending on MPO compression) 
        print("Average estimation error for {} perturbations: {:.3f}".format(
            perturbations, np.real(np.mean(error))))
    else:
        # plot errors as histogram
        n, bins, patches = plt.hist(x=error, bins=len(
            error)//10, alpha=0.65, color='red', histtype='step')
        plt.xlabel("Error")
        plt.ylabel("Counts")
        plt.title("Error distribution for {} qubit Clifford+T unitary".format(N//2))
        plt.show()

if __name__ == '__main__':
    # generate interesting unitary and its stabilisers
    U = mporep(mpo_dict["id"], mpo_dict["id"], 15)

    # estimate fidelity of U when perturbed five thousand times using finite sampling of outcome distributions
    UniU_error(U=U, N=16, perturbations=20000, exact=False)
