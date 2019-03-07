from EnhancedDCM.utilities import grad_hess_utilities as ghu
from keras.models import load_model
from generated_data import data_manager as dm
import numpy as np
from keras import backend as K


def fetch_model(neuron, path, extend):
	""" Load models of generated (synthetic) run """
	filename = "{}generated_{}{}.h5".format(path, neuron, extend)
	return load_model(filename)


def get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=False, lmnlArchitecture=False, trueArchitecture=False, write=False):
	""" Get model inputs for each .dat for Train and Test """
	inputs_labels, extra, _ = dm.keras_input(filePath, fileInputName, filePart, simpleArchitecture=simpleArchitecture,
											 lmnlArchitecture=lmnlArchitecture, trueArchitecture = trueArchitecture, write=write)
	labels = inputs_labels[:,-1,:]
	inputs = np.delete(inputs_labels, -1, axis = 1)
	inputs = np.expand_dims(inputs, -1)

	extra = np.expand_dims(extra,-1)
	extra = np.expand_dims(extra,-1)
	extra = (extra - extra.mean(axis=0)) / extra.std(axis=0)

	return inputs, extra, labels


def scan_iteration(number, cases, paths_extends, filePath, fileInputBase, encyclopedia, std_flag=False):
	"""
	Fills a dictionary with all important model statistics
	:param number: 			Dictionary index (e.g. # M.C. experiment)
    :param cases:           model name type
    :param paths_extends:   tuple with path to model and model saveName extension
    :param filePath:        labels filepath
    :param fileInputBase:   base name of labels and models
    :param std_flag:        when active, adds stds from Hessian when possible (slower process)
	:param enyclopedia:   dictionary of model values
	:return; encyclopedia
	"""
	fileInputName = fileInputBase +'_{}'.format(number)

	def get_model_inputs_labels(filePart):
		"""Get model inputs for each .dat for Train and Test"""
		inputs, extra_input, labels = get_inputs_labels(filePath, fileInputName, filePart)
		inputs_simple, extra_input_simple, _ = get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=True)
		inputs_lmnl, extra_input_lmnl, _ = get_inputs_labels(filePath, fileInputName, filePart, lmnlArchitecture=True)
		inputs_true, extra_input_true, _ = get_inputs_labels(filePath, fileInputName, filePart, trueArchitecture=True)
		# Assign inputs to models 
		model_inputs = { case:[inputs] for case in cases if 'FULL' in case}
		model_inputs.update({ case:[inputs_simple] for case in cases if 'FULL' not in case})
		# MNL models have single input in Input layer
		model_inputs['HYBRID'] = [inputs_lmnl, extra_input_lmnl]
		model_inputs['HRUSCHKA2004'] = [inputs]
		model_inputs['TRUE_MNL'] = [inputs_true]
		return model_inputs, labels

	model_inputs, train_labels = get_model_inputs_labels('_train')
	model_test_inputs, test_labels = get_model_inputs_labels('_test')

	#Dict to save all values and plotting
	encyclopedia[number] = {}
	for case in cases:
		NN_flag = case == 'HRUSCHKA2004' 
		# Get model extension name and path to .dat file
		path, extend = paths_extends[case]
		model = fetch_model(number, path, extend)
		if not NN_flag:
			betas = ghu.get_betas(model)
		likelihood_train, accuracy_train = ghu.get_likelihood_accuracy(model, model_inputs[case], train_labels)
		likelihood_test, accuracy_test = ghu.get_likelihood_accuracy(model, model_test_inputs[case], test_labels)
		# Getting STD is slow (1-3s), avoid if possible
		if std_flag and not NN_flag:
			stds = ghu.get_stds(model, model_inputs[case], train_labels)
		if not NN_flag:
			encyclopedia[number]['betas_'+case] = betas
		encyclopedia[number]['likelihood_train_'+case] = likelihood_train
		encyclopedia[number]['likelihood_test_'+case] = likelihood_test
		encyclopedia[number]['accuracy_train_'+case] = accuracy_train
		encyclopedia[number]['accuracy_test_'+case] = accuracy_test
		if std_flag and not NN_flag:
			encyclopedia[number]['stds_'+case] = stds

	encyclopedia[number]['coef'] = np.load(paths_extends['coef']+'coef_{}.npy'.format(number))
	K.clear_session()
	return encyclopedia[number]