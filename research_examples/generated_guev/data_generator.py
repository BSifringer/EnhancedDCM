import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

""" Synthetic Data Generator

	Parameters:
	---------
	Flags:
		- noise: Sets non-linear parameters to 0, hence giving noise to the NN component
		- correlations: Changes x3' to be correlated with x1. p values folllow plist
		- unseen: Adds causalities to the Utility
	Variables:
		- n_files: # of experiments
		- n_samples: # samples per experiment
		- plist: correlation coefficients. n_files experiments are done for every p of plist
"""

np.random.seed(0)


parser = argparse.ArgumentParser(description='Choose a Flag for experiment')

# Flags:
parser.add_argument('--illustrate', action='store_true', help='Defines Betas to specific values instead of random sampling')
parser.add_argument('--correlations', action='store_true', help='Multiple datasets created with defined correlation between generated variables')
parser.add_argument('--unseen', action='store_true', help='Adds variables to the utility function for unseen causality experiment')
parser.add_argument('--noise', action='store_true', help='Sets non-linear parameters to 0, hence giving noise to the NN component')

# Params:
parser.add_argument('--n_files', type=int, default=1, help='# of experiments, (or utility functions)')
parser.add_argument('--n_samples', type=int, default=10000, help='# of generated samples per utility function')
parser.add_argument('--folder', type=str, default='', help='Store datasets in defined folder')

args = parser.parse_args()


filePath = os.path.dirname(os.path.realpath(__file__))+'/'
folderName = args.folder

noise = args.noise
correlations = args.correlations
p_list = [1, 0.95, 0.9, 0.85,  0.8, 0.6, 0.4, 0.2, 0.0]
unseen = args.unseen
illustrate = args.illustrate

n_files = args.n_files
n_samples = args.n_samples
n_args = 4

def invlogit(x):
	return np.exp(x) / (1 + np.exp(x))

def ProbaX(x,y):
	return np.exp(x) / (np.exp(x) + np.exp(y))

def generate_outcomes(n, coeff, *args):
	A1, A2, A3, A4, *_ = args
	l=1
	a1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	a2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	b1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	b2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	h1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	h2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	ek1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	ek2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	eq1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	eq2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	ep1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	ep2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)

	z1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	z2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	wz1 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
	wz2 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)

	k1 = 1*h1 + ek1
	k2 = 1*h2 + ek2

	q1 = 2*h1 + 1*k1  + eq1
	q2 = 2*h2 + 1*k2  + eq2

	p1 =  5+ z1+ 0.03*wz1 + ep1  #+ 1.0*q1
	p2 =  5+ z2+ 0.03*wz2 + ep2  #+ 1.0*q2

	c1 = np.expand_dims(np.random.uniform(low=-1,high=1,size=n),1)
	c2 = np.expand_dims(np.random.uniform(low=-1,high=1,size=n),1)




	if correlations:
		p1 = coeff*q1 + p1*(1-coeff**2)**0.5
		p2 = coeff*q2 + p2*(1-coeff**2)**0.5

	if unseen:
		x6 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
		x7 = np.expand_dims(np.random.uniform(low=-l,high=l,size=n),1)
		A5 = _[0]
# ------------------- The Toy Function ------------------------------
	U1 = A1*p1 + A2*a1 + A3*b1+ A4*q1*c1 #+ ev1 instead of gumbel distribution, we perform invert logit probability sampling
	U2 = A1*p2 + A2*a2 + A3*b2+ A4*q2*c2 #+ ev2
# -------------------------------------------------------------------
	if unseen:
		U1 = U1 + A5*x6
		U2 = U2 + A5*x7
		return np.random.binomial(1, ProbaX(U1,U2)), p1,p2, a1,a2, b1,b2, q1,q2, c1,c2, x6,x7

	return np.random.binomial(1, ProbaX(U1,U2)), p1,p2, a1,a2, b1,b2, q1,q2, c1,c2
	# return ProbaX(U1,U2),p1,p2, a1,a2, b1,b2, q1,q2, c1,c2


def saveFile(fileName, data, headers):
	file = open(fileName, 'wb')
	np.savetxt(file, data, fmt='%10.5f', header=headers, delimiter = '\t', comments='')
	file.close()


def single_run(n, i, coeff, *args):
	"""
	Creates and saves a dataset on generated outcomes
	input:
		- n: size of dataset
		- a: function coefficients
		- i: index of dataset
	"""
	headers = "p1\tp2\ta1\ta2\tb1\tb2\tq1\tq2\tc1\tc2\tchoice"
	if unseen:
		headers = "p1\tp2\ta1\ta2\tb1\tb2\tq1\tq2\tc1\tc2\tx6\tx7\tchoice"

	outcomes,  p1,p2, a1,a2, b1,b2, q1,q2, c1,c2, *_ = generate_outcomes(n, coeff, *args)
	data = np.concatenate(( p1,p2, a1,a2, b1,b2, q1,q2, c1,c2,*_, outcomes), axis = 1)
	saveFile(filePath+ folderName + 'generated_{}_train.dat'.format(i), data, headers)

	n_test = int(n*0.2)
	outcomes_test,  p1,p2, a1,a2, b1,b2, q1,q2, c1,c2, *_ = generate_outcomes(n_test, coeff, *args)
	data_test = np.concatenate(( p1,p2, a1,a2, b1,b2, q1,q2, c1,c2,*_, outcomes_test), axis = 1)
	saveFile(filePath+ folderName + 'generated_{}_test.dat'.format(i), data_test, headers)

	return data, data_test

if __name__ == "__main__" :
	if unseen:
		n_args = n_args+1
	if not correlations:
		p_list = [1]
	total = np.zeros(3)
	for i, coeff in enumerate(p_list):
		for i in range(i*n_files, (i+1)*n_files):
			args = np.random.rand(n_args)*9 - 4.5
			args = [arg+0.5 if arg>=0 else arg for arg in args]
			args = [arg-0.5 if arg<0 else arg for arg in args]

			if noise:
				args[2:4] = [0, 0, 0]

			if illustrate:
				args = [-2, 1, 0.5, 1]
			train, test = single_run(n_samples, i, coeff, *args)
			np.save(filePath + folderName + 'coef_{}.npy'.format(i), args)
			total = np.concatenate([total, train[:,-1]])
		# plt.hist(total)
		# plt.show()
