import mpnum as mp
import numpy as np
from qinfo import *
from operators import *
from scipy.linalg import expm
from itertools import product, combinations
import matplotlib.pyplot as plt



######
# measurements
N = int(5e3)
# true unitary to verify
U = np.kron(op1['id'],np.eye(2))
# rotation operator
alpha = 0.2
Urotx = expm(-1j*np.pi*alpha*op1['x']/2)
Uroty = expm(-1j*np.pi*alpha*op1['y']/2)
Urotz = expm(-1j*np.pi*alpha*op1['z']/2)
# whether to use the random seed
seed = False
######

# set random seed for repeatable runs
if seed:
    np.random.seed(347862)

def rand_res(probs=[0.5,0.5], outcomes=[0,1], num=1):
    """
    Returns a set of <num> <outcomes> using the probability weighting <probs> given.
    """
    # sanitise input probabilities (floating point error can lead to negative/complex values
    probs = np.clip(np.real(probs),0,1)
    return np.random.choice(outcomes,size=(1,num),p=probs)[0]

def mle(xi):
    """
    Returns the estimate for the angle
    """
    return np.sum(xi)/len(xi)


def angle(qwr):
    """
    returns the value of 
    """
    return np.arccos(qwr-2)/np.pi


def var(theta, esum, n):
    """
    estimates the variance of the parameter
    """
    H = -esum/theta**2 - (n-esum)/((1-theta)**2) 
    return (-H)**(-1)




# define measurement operators
I = np.eye(2)
X = op1['x']
Y = op1['y']
Z = op1['z']
XX = np.kron(op1['x'] , op1['x'])
YY = np.kron(op1['y'] , op1['y'])
ZZ = np.kron(op1['z'], op1['z'])
ZI = np.kron(op1['z'], op1['z'])

# same for 4 qubits
XXXX = np.kron(XX,XX)
ZZZZ = np.kron(ZZ,ZZ)
ZIZI = np.kron(np.kron(Z,I),np.kron(Z,I))
ZIIZ = np.kron(np.kron(Z,I),np.kron(I,Z))

# define projectors
Q = (np.eye(4)-U @ XX @ dagger(U))/2
W = (np.eye(4)-U @ YY @ dagger(U))/2
R = (np.eye(4)-U @ ZZ @ dagger(U))/2

def eye_like(m):
    """
    Returns identity matrix with same dims as square matrix m.
    """
    dim = np.shape(m)[0]
    return np.eye(dim)


def exp_val_av(stabiliser, state, urot):
    """
    compute average expectation value
    """
    s = 0
    for r in stabiliser:
        s += np.trace((np.identity(16)+r) @ (urot @ state @ dagger(urot)))/2

    print(s)


def prob_calc(projector, state):
    """
    Compute the outcome probability for the given state and projector
    """
    return np.trace(projector @ state)

def theta_compute(stab_sum, N):
    """
    Compute the angle of rotation for an N qubit estimator.
    """
    modi = 2**(N/2+1)-1-2**(N/2-1)
    if stab_sum < 2**(N/2+1)-1-2**(N/2-1)*2: 
        stab_sum = 2**(N/2+1)-1-2**(N/2-1)*2

    return np.arccos((np.round(stab_sum,5)-modi)/2**(N/2-1))*0.5

def angle_estimate(stabilisers, state, N, shots=500):
    """
    Compute Bures angle given the perturbed state and the stabilisers for the True state
    """

    # initialise total 
    stab_sum = 0.0
    # iterate over stabilisers
    for stab in stabilisers:
        # compute out probability
        prob = prob_calc((np.eye(2**N)+stab)/2, state)
        stab_sum += mle(rand_res([1-prob,prob], num=int(shots)))

    theta = theta_compute(stab_sum, N=N)
    uncert = 0 #np.sqrt(var(theta, stab_sum, int(N)))

    return theta,uncert


def bures_est(alpha_est, state):
    """
    Compute the Bures estimate of the state
    """

    U = np.kron(expm(-1j*np.pi*alpha_est*op1['x']/2),np.eye(8))

    state_est = U @ state @ dagger(U)

    return bures(state, state_est)

def random_Ugen(object):
    """
    generator that takes array of random unitaries as input and returns a generator object
    producing matrix product arrays.
    """
    def __init__(self, Us, N):
        # fucking really? What is the point?
        self.Us = Us
        # total number of unitaries
        self.num = np.shape(Us)[2]
        # current iteration
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """
        Compute next matrix product operator
        """
        if self.n < self.num:
            mpo_rand = mp.MPArray.from_array_global(self.Us[:,:,self.n].reshape([2]*2**N), ndims=2)
            self.n += 1
            return mpo_rand

        # end generation
        else:
            raise StopIteration


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
    mps = mp.MPArray.from_kron([np.array([1,0])]*N)

    # generate entanglement operator in MPO form
    hadamard_mpo = mporep(mpo_dict["h"], mpo_dict["id"], reps=N-1)
    cx_mpo = mp.chain([mpo_dict["id"]]*N)
    for i in range(0,N-1):
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
    assert N%2==0, "Bell states must have even number of qubits" 

    # default XXX...X generator

    generators = [mpojob([mpo_dict['x']], [0]*N)]

    # iterate over maximum number of generators needed (Lemma 2)
    for i in range(1, int(N/2) + 1):
        # placement vector
        selector = list([0]*N)
        # select Z_n otimes Z_n-1
        selector[i-1:i+1] = (1,1)
        # construct stabiliser generator
        new_gen = mpojob([mpo_dict["id"], mpo_dict["z"]], selector)
        # add to set
        generators.append(new_gen)

    return generators

def base_gen(N=2):
    """
    Generates a stabiliser input basis for |0>^n
    """
    def rotate(l, n):
        return l[-n:] + l[:-n]

    bid = ['I']*N
    bid[0] = 'Z'
    base = [bid]
    for i in range(1,N):
        base.append(rotate(bid,i))

    return base


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

        # generate every unique 
        gen_combs = set(list(combinations(indices, i)))

        # iterate over combinations and add to 
        for comb in gen_combs:
            
            # create new tblet
            stab = mpojob([mpo_dict["id"]], [0]*N)
            for k in comb: 

                # add generator to stabiliser sequence
                stab = mp.dot(gen[k], stab)

            # add new stabiliser to set
            stabilisers.append(stab)



    return stabilisers



def UniU_error(perturbations=100, exact=True):
    """
    Check error rates on verification technique
    """
    N = 24
    
    # generate entangled states
    rho = bell_gen(N=N)

    # generate Bell state stabilisers
    bell_gstab = bell_stab_gen(N=N)

    # generate interesting unitary and its stabilisers 
    Uni_U = mporep(mpo_dict["s"], mpo_dict["id"], N-1) #np.kron(Universal_U(), np.eye(2**(N//2)))

    # apply to entangled state
    rho = mp.dot(Uni_U, rho)

    # evolve generators under unitary
    gstab = [mp.dot(mp.dot(Uni_U, stb), Uni_U.adj()) for stb in bell_gstab]

    # generate stabiliser set 
    stabilisers = operator_find(gstab, N=N)

    # apply to entangled state and convert to MPO for measurement phase
    rho = mp.mpsmpo.mps_to_mpo(rho)

    # generate unitary perturbations
    #Us = random_U(N=N//2, num=states)

    # convert to matrix product states generator
    #Us = random_Ugen(Us, N=N//2)

    # calculate the estimation error
    error = []
    for i in range(0, perturbations):
        print("Now computing unitary perturbation {}".format(i))

        # copy state (why?)
        rho_p = rho.copy()

        # compute a local perturbation - non-compliant with MPS right now
        #U_p = np.kron(Us[:,:,i], np.eye(2**(N//2)))
        #rho_p = mp.dot(U_p, rho_c)


        # compute expectation values exactly or with finite samples
        if exact:
            Q = 0.0
            # iterate over stabiliser measurements
            for stab_proj in stabilisers:
                # add to Q sum
                Q += (1 + mp.trace(mp.dot(stab_proj, rho_p)))/2
            # estimate angle 
            a_est = theta_compute(Q, N=N)

        else:
            # estimate expectation values from finite number of outcomes
            a_est, a_uncert = angle_estimate(stabilisers, rho_p, N=N, shots=8000)

            if np.abs(np.real(bures(rho_c, rho_p) - a_est)) > 0.5:
                print(a_est)
                continue

        # compute angle estimate error
        error.append(a_est)
        #error.append(np.real(bures(rho, rho_p) - a_est))

    if exact:
        print("Average estimation error for {} perturbations: {:.3f}".format(perturbations, np.mean(error)))
    else:
        n,bins,patches = plt.hist(x=error, bins=len(error)//10, alpha=0.65, color='red', histtype='step')
        plt.xlabel("Error")
        plt.ylabel("Counts")
        plt.title("Error distribution for {} qubit Clifford+T unitary".format(N//2))
        plt.show()


UniU_error(perturbations=5, exact=True)
exit()

# define singlet state
rho_s = (np.eye(4)-XX-YY-ZZ)/4

# create perturbed unitary
Upert = np.kron(Urotx,np.eye(2)) @ U

# generate 4 qubit entangled state
rho4 = bell_gen(N=4)


error = []
stabilisers = operator_find([XXXX,ZZZZ,ZIIZ], n=6)
print(len(stabilisers))
exit()

for i in range(0, 10000):
    alpha1 = np.random.ranf()*0.5
    alpha2 = np.random.ranf()*0.5

    Urot1 = expm(-1j*np.pi*alpha1*op1['x']/2)
    Urot2 = expm(-1j*np.pi*alpha2*op1['y']/2)


    Urot = np.kron(np.kron(Urot1,Urot2),np.eye(4))
    state = Urot @ rho4 @ dagger(Urot)
    
    a_est,a_uncert = angle_estimate(stabilisers, state)
    b_est = bures_est(a_est, rho4)

    error.append(np.real(bures(rho4, state) - b_est))



# plot error distribution
n,bins,patches = plt.hist(x=error, bins=len(error)//10, alpha=0.65, color='red', histtype='step')
plt.xlabel("Error")
plt.ylabel("Counts")
plt.title("Error distribution for two qubit process with {} shots".format(N))
plt.show()

print("Actual Bures angle: {:.4f}".format(bures(rho4, state)))
print("Estimate of Bures angle: {:.4f} +/- {:.4f}".format(b_est, a_uncert))



# define perturbed state
rho_p = Upert @ rho_s @ dagger(Upert)

# outcome average for each projector
prob = np.cos(np.pi*alpha/2)**2

Qe = mle(rand_res(probs=[0,1],num=int(N)))
We = mle(rand_res(probs=[1-prob,prob],num=int(N)))
Re = mle(rand_res(probs=[1-prob,prob],num=int(N)))

theta = angle(Qe+We+Re)

