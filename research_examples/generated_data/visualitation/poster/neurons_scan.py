import numpy as np
import pickle
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
	""" Imports fix, Call function in it's directory """
	from sys import path
	from os.path import dirname as dir
	path.append(dir(path[0]))
	splits = path[0].split('/')

	parent = '/'.join(splits[:-4])
	path.append(dir(parent))
	parent = '/'.join(splits[:-3])
	path.append(dir(parent))
	parent = '/'.join(splits[:-2])
	path.append(dir(parent))
	parent = '/'.join(splits[:-1])
	path.append(dir(parent))

	__package__ = "generated_data"

from EnhancedDCM.utilities import grad_hess_utilities as ghu
from generated_data import data_manager as dm
from keras.models import load_model

"""
	Create dictionnary for Synthetic Data L-MNL Neuron scan
"""

def fetch_model(neuron, path, extend):
	""" Load models from synthetic Neuron Scan """
	filename = "{}generated_data_Enhancedscan{}{}.h5".format(path, neuron, extend)
	filename = "{}generated_0_Enhancedscan{}{}.h5".format(path, neuron, extend)
	return load_model(filename)

def get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=False, lmnlArchitecture=False, write=False):
	""" Get model inputs for each .dat for Train and Test """
	inputs_labels, extra, _ = dm.keras_input(filePath, fileInputName, filePart, simpleArchitecture=simpleArchitecture,
											 lmnlArchitecture=lmnlArchitecture, write=write)

	labels = inputs_labels[:,-1,:]
	inputs = np.delete(inputs_labels, -1, axis = 1)
	inputs = np.expand_dims(inputs, -1)

	extra = np.expand_dims(extra,-1)
	extra = np.expand_dims(extra,-1)
	extra = (extra - extra.mean(axis=0)) / extra.std(axis=0)

	return [inputs, extra], labels


if __name__ == '__main__':

	path, extend = ['../../poster/','extra']
	path, extend = ['../../illustrate/','extra']
	encyclopedia = {}

	#model_inputs, train_labels = get_inputs_labels(path, 'generated_data','_train', lmnlArchitecture=True)
	#model_test_inputs, test_labels = get_inputs_labels(path,'generated_data', '_test', lmnlArchitecture=True)
	model_inputs, train_labels = get_inputs_labels(path, 'generated_0','_train', lmnlArchitecture=True)
	model_test_inputs, test_labels = get_inputs_labels(path,'generated_0', '_test', lmnlArchitecture=True)

	list = [2,5,10, 15, 25, 50, 100, 200, 500, 1001, 2000]
	#list = [15]
	for number in tqdm(list):

		encyclopedia[number] = {}
		model = fetch_model(number, path, extend)
		betas = ghu.get_betas(model)

		likelihood_train, accuracy_train = ghu.get_likelihood_accuracy(model, model_inputs, train_labels)
		likelihood_test, accuracy_test = ghu.get_likelihood_accuracy(model, model_test_inputs, test_labels)

		encyclopedia[number]['betas'] = betas
		encyclopedia[number]['likelihood_train'] = likelihood_train
		encyclopedia[number]['likelihood_test'] = likelihood_test
		encyclopedia[number]['accuracy_train'] = accuracy_train
		encyclopedia[number]['accuracy_test'] = accuracy_test
		
	pickle.dump(encyclopedia, open('encyclo_mult.p', 'wb'))