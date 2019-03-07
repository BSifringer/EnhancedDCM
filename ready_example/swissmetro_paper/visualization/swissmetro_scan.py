import numpy as np
import pickle

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    print(path[0])
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-2])
    path.append(dir(parent))
    parent = '/'.join(splits[:-3])
    path.append(dir(parent))
    parent = '/'.join(splits[:-1])
    path.append(dir(parent))

    __package__ = "generated_data"

from EnhancedDCM.utilities import grad_hess_utilities as ghu
from swissmetro_paper import data_manager as dm
from keras.models import load_model


"""
	Saves results of Swissmetro L-MNL neuron scan models
"""


def fetch_model(neuron, path, extend):
	filename = "{}swissmetro_Enhanced{}{}.h5".format(path, neuron, extend)
	return load_model(filename)

def get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=False, write=False):
	inputs_labels, extra, _ = dm.keras_input(filePath, fileInputName, filePart, simpleArchitecture=simpleArchitecture, write=write)

	labels = inputs_labels[:,-1,:]
	inputs = np.delete(inputs_labels, -1, axis = 1)
	inputs = np.expand_dims(inputs, -1)

	extra = np.expand_dims(extra,-1)
	extra = np.expand_dims(extra,-1)
	extra = (extra-extra.mean(axis=0))/extra.std(axis=0)

	return [inputs, extra], labels


if __name__ == '__main__':

	path, extend = ['../scan/','extra']
	encyclopedia = {}

	model_inputs, train_labels = get_inputs_labels(path, 'swissmetro','_train', simpleArchitecture = True)
	model_test_inputs, test_labels = get_inputs_labels(path,'swissmetro', '_test', simpleArchitecture = True)

	list = [1,5,10, 15, 25, 50, 100, 200, 500, 1001, 2000, 5000]
	for number in list:

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
