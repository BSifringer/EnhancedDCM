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

def generate_outcomes(n, coeff, *args):
	a1, a2, a3, a4, *_ = args

	x1 = np.expand_dims(np.random.normal(size=n),1)
	x2 = np.expand_dims(np.random.normal(size=n),1)
	x3 = np.expand_dims(np.random.normal(size=n),1)
	x4 = np.expand_dims(np.random.normal(size=n),1)
	x5 = np.expand_dims(np.random.normal(size=n),1)
	if correlations:
		x3 = coeff*x1 + x3*(1-coeff**2)**0.5 
	if unseen:
		x6 = np.expand_dims(np.random.normal(size=n),1)
		x7 = np.expand_dims(np.random.normal(size=n),1)
		a5 = _[0]
# ------------------- The Toy Function ------------------------------
	predictors = a1*x1 + a2*x2 + a3*x3*x4 + a4*x3*x5
	predictors = 2*x1 + 3*x2 + 0.5*x3*x4 + 1*x3*x5

# -------------------------------------------------------------------
	if unseen:
		predictors = predictors + a5*x6*x7
		return np.random.binomial(1, invlogit(predictors)), x1, x2, x3, x4, x5, x6, x7

	return np.random.binomial(1, invlogit(predictors)), x1, x2, x3, x4, x5
	#return invlogit(predictors), x1, x2, x3, x4, x5


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
	headers = "x1\tx2\tx3\tx4\tx5\tchoice"
	if unseen:
		headers="x1\tx2\tx3\tx4\tx5\tx6\tx7\tchoice"

	outcomes, x1, x2, x3, x4, x5, *_ = generate_outcomes(n, coeff, *args)		
	data = np.concatenate((x1,x2,x3,x4,x5,*_, outcomes), axis = 1)
	saveFile(filePath+ folderName + 'generated_{}_train.dat'.format(i), data, headers)
	
	n_test = int(n*0.2)	
	outcomes_test, x1, x2, x3, x4, x5, *_ = generate_outcomes(n_test, coeff, *args)
	data_test = np.concatenate((x1,x2,x3,x4,x5,*_, outcomes_test), axis = 1)
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
				args = [2, 3, 0.5, 1]
			train, test = single_run(n_samples, i, coeff, *args)
			np.save(filePath + folderName + 'coef_{}.npy'.format(i), args)
			total = np.concatenate([total, train[:,-1]])
		#plt.hist(total)
		#plt.show()
