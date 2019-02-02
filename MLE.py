import numpy as np
from qinfo import *
from operators import *
from scipy.linalg import expm
from itertools import product
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



def operator_find(gen, depth=1):
	"""
	Brute forces calculation of generator products (crude but whatever)
	"""
	str_ind = [str(i) for i in list(range(0,len(gen)))]
	
	# compute all combinations with repeats of length n
	stabilisers = []
	for k in range(1,depth+1):
		combs = product(str_ind, repeat=k)
		for comb in combs:
			# compute new array
			a = np.asarray(gen[0] @ dagger(gen[0]))
			for g in comb:
				a = gen[int(g)] @ a

			# check if array is the identity
			if np.allclose(a, np.eye(len(gen[0]))):
				continue

			if not exist_check(stabilisers, a):   
				stabilisers.append(a)

	return stabilisers
			

def exist_check(collect, a):
	"""
	Check if a is in collection
	"""
	for i in collect:
		if np.allclose(i,a):
			return True
	return False


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
	uncert = np.sqrt(var(theta, stab_sum, int(N)))

	return theta,uncert


def bures_est(alpha_est, state):
	"""
	Compute the Bures estimate of the state
	"""

	U = np.kron(expm(-1j*np.pi*alpha_est*op1['x']/2),np.eye(8))

	state_est = U @ state @ dagger(U)

	return bures(state, state_est)



def stab_gen(U,base):
	"""
	Computes the set of stabilisers given an input basis and
	description of a clifford unitary.
	base = [['Z','I','I','I','I'],['I','Z'...]]
	U = [('H',1), ('CX',1,2), ...]

	"""
	# initialise updated stabiliser base
	ubase = np.zeros_like(base)
	for g,b in enumerate(base):
		for gate in U:
			if gate[0]=='H':
				if b[gate[1]-1]=='I': pass
				elif b[gate[1]-1]=='Z': b[gate[1]-1] = 'X'
				elif b[gate[1]-1]=='X': b[gate[1]-1] = 'Z'

			elif gate[0]=='CX':
				# extract important ops
				ctrl = b[gate[1]-1]
				trgt = b[gate[2]-1]

				# propagate ctrl
				if ctrl=='I' or ctrl=='Z':
					# commutes with control
					pass
				elif ctrl=='X':
					if trgt=='X': b[gate[2]-1]='I'
					elif trgt=='I': b[gate[2]-1]='X'
					elif trgt=='Z': b[gate[2]-1]='Y'

				# propagate trgt
				if trgt=='I' or trgt=='X':
					# commutes with target
					pass
				elif trgt == 'Z':
					if ctrl=='X': b[gate[1]-1]='Y'
					elif ctrl=='I': b[gate[1]-1]='Z'
					elif ctrl=='Z': b[gate[1]-1]='I'
		ubase[g] = b

	return ubase


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


def UniU_error(states=100):
	"""
	Check
	"""
	rho8 = bell_gen(N=8)

	# generate interesting unitary and its stabilisers 
	Uni_U = np.kron(Universal_U(), np.eye(2**4))

	# apply to entangled state
	rho = Uni_U @ rho8 @ dagger(Uni_U)

	# generate stabilisers (this was painful should automate)
	stabs = []
	stabs.append(-1*kronjob([op1['id'],op1['x'],op1['y'],op1['z']],[0,0,1,0,1,1,1,1])) # XXXX XXXX

	stabs.append(kronjob([op1['id'],op1['x'],op1['y'],op1['z']],[2,0,0,0,0,0,0,0])) # ZZII IIII

	stab = kronjob([op1['id'],op1['x'],op1['y'],op1['z'],op1['r']],[2,4,1,0,0,0,0,0]) # ZIZI IIII
	stabs.append(-1j*kronjob([op1['id'],op1['x'],op1['y'],op1['z'],op1['r']],[0,3,0,0,0,0,0,0]) @ stab) 

	stab = kronjob([op1['id'],op1['x'],op1['y'],op1['z'],op1['r']],[2,4,1,3,0,0,0,0]) # ZIIZ IIII
	stabs.append(-1j*kronjob([op1['id'],op1['x'],op1['y'],op1['z'],op1['r']],[0,3,0,0,0,0,0,0]) @ stab) 

	stabs.append(kronjob([op1['id'],op1['x'],op1['y'],op1['z'],op1['r']],[2,4,2,0,3,0,0,0])) # ZIII ZIII

	# generate stabiliser set
	stabilisers = operator_find(stabs, depth=5)

	# apply to entangled state
	rho = Uni_U @ rho8 @ dagger(Uni_U)

	# generate unitary perturbations
	Us = random_U(N=4, num=states)

	# calculate the estimation error
	error = []
	for i in range(0,states):
		if i % 250 == 0: print(i)
		# copy state (why?)
		rho_c = np.copy(rho)

		# compute a local perturbation
		U_p = np.kron(Us[:,:,i], np.eye(2**4))
		rho_p = U_p @ rho_c @ dagger(U_p)

		# estimate expectation values from finite number of outcomes
		a_est, a_uncert = angle_estimate(stabilisers, rho_p, N=8, shots=5000)

		if np.abs(np.real(bures(rho_c, rho_p) - a_est)) > 0.5:
			print(a_est)
			continue

		# compute angle estimate error
		error.append(np.real(bures(rho_c, rho_p) - a_est))

	n,bins,patches = plt.hist(x=error, bins=len(error)//10, alpha=0.65, color='red', histtype='step')
	plt.xlabel("Error")
	plt.ylabel("Counts")
	plt.title("Error distribution for 4 qubit Clifford+T unitary")
	plt.show()


UniU_error(states=500)
exit()

# define singlet state
rho_s = (np.eye(4)-XX-YY-ZZ)/4

# create perturbed unitary
Upert = np.kron(Urotx,np.eye(2)) @ U

# define 4 qubit entangled state (mathematica transplant)
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

