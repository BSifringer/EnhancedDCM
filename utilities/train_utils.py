import numpy as np
import tensorflow as tf
import random
import shelve
import _pickle as pickle
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping,  ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import np_utils, plot_model
from keras.losses import mean_squared_error



"""	Utilities for Training:

	- Loads the data Given a full path to the .npy
	- Trains a Compiled Model (single input or two inputs) from Run Utilities

	Note: X input data not normalized (e.g. to preserve coherence/interpretability (cost in chf, time in min, etc..)
		  Q input data (=extra) normalized before training
"""


def train_sameInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '', save_model = True, train_val_indices = None,
					saveExtension = '', filePart = '', callback = None, validationRatio = 0, Hrusch=False , verbose = 0, model_inputs = None):

	if model_inputs is not None:
		train_data = model_inputs
	else:
		train_data = np.load(train_data_name)

	train_labels = train_data[:,-1,:]
	train_data = np.delete(train_data, -1, axis = 1)

	train_data = np.expand_dims(train_data, -1)
#	train_data = normalize(train_data)

	if not Hrusch:
		history = fitModel(train_data, train_labels, nEpoch,batchSize,compiledModel,callback,validationRatio, train_val_indices, verbose=verbose)
	else:
		#Each Utility has its own input layer to share the first dense network
		history = fitModel([np.expand_dims(train_data[:,:,i],-1) for i in range(train_data.shape[2])], train_labels,
						   nEpoch,batchSize,compiledModel,callback,validationRatio, train_val_indices, verbose=verbose)

	#with open(filePath + 'trainHistoryDict' + saveExtension, 'wb') as file_pi:
	#	pickle.dump(history.history, file_pi)
	if save_model:
		compiledModel.save(filePath + fileInputName +'_' + saveExtension + '.h5')

	return history

def train_extraInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '', save_model = True, save_weights = False, train_val_indices = None, e_cancel = False,
					 saveExtension = '', filePart = '', callback = None, validationRatio = 0, NN = False , verbose = 0, model_inputs = None, e_normalize = False):

	if model_inputs is not None:
		train_data = model_inputs[0]
		extra_data = model_inputs[1]
	else:
		train_data = np.load(train_data_name)
		extra_data = np.load(train_data_name[:-4] + '_extra.npy')

	train_labels = train_data[:,-1,:]
	train_data = np.delete(train_data, -1, axis = 1)
	train_data = np.expand_dims(train_data, -1)

	nExtraFeatures = extra_data[0].size
	extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

	#train_data = normalize(train_data)
	if e_normalize:
		extra_data = normalize(extra_data)
	if e_cancel:
		extra_data = extra_data * 0

	if not NN:
		history = fitModel([train_data, extra_data], train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, train_val_indices, verbose = verbose)
	else:
		history = fitModel(extra_data, train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, train_val_indices, verbose = verbose)

	#with open(filePath + 'trainHistoryDict' + saveExtension, 'wb') as file_pi:
	#	pickle.dump(history.history, file_pi)
	if save_model:
		compiledModel.save(filePath +fileInputName + '_' + saveExtension + '.h5')
	#	compiledModel.save_weights(filePath +fileInputName + '_' + saveExtension + '.h5')
	if save_weights:
		compiledModel.save_weights(filePath +fileInputName + '_' + saveExtension + '.h5')

	return history

def train_subInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '', save_model = True, train_val_indices = None,
					 saveExtension = '', filePart = '', callback = None, validationRatio = 0, NN = False , verbose = 0, model_inputs = None, e_normalize = False):

	if model_inputs is not None:
		train_data = model_inputs[0]
		extra_data = model_inputs[1:]
	else:
		train_data = np.load(train_data_name)
		extra_data = np.load(train_data_name[:-4] + '_extra.npy')
		extra_data = [subset for subset in extra_data]

	train_labels = train_data[:,-1,:]
	train_data = np.delete(train_data, -1, axis=1)
	train_data = np.expand_dims(train_data, -1)

	nExtraFeatures = extra_data[0][0].size
	extra_data = [np.expand_dims(np.expand_dims(subset, -1),-1) for subset in extra_data]

	#train_data = normalize(train_data)
	if e_normalize:
		extra_data = [normalize(subset) for subset in extra_data]

	history = fitModel([train_data, *extra_data], train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, train_val_indices, verbose = verbose)
	#with open(filePath + 'trainHistoryDict' + saveExtension, 'wb') as file_pi:
	#	pickle.dump(history.history, file_pi)
	if save_model:
		compiledModel.save(filePath +fileInputName + '_' + saveExtension + '.h5')

	return history

def fitModel(train_data, train_labels, nEpoch, batchSize, model, callback, validationRatio, train_val_indices, verbose = 0):
	""" Call fit function. Different calls whether use of Callback or validationRatio"""
    #plateau = ReduceLROnPlateau(patience = 20, verbose=1, factor = 0.8)
	args = {'x': train_data,
			'y': train_labels,
            # 'batch_size' : batchSize,
			'steps_per_epoch' : batchSize,
			'epochs' : nEpoch,
			'verbose': verbose
            }

	if callback is not None:
		args['callbacks'] = callback

	if validationRatio != 0 or train_val_indices is not None:
		args['validation_split'] = validationRatio
		args['validation_steps'] = batchSize

	if train_val_indices is not None:
		if len(train_data) <=2:
			args.update({
				'x': [train_data[0][train_val_indices[0]], train_data[1][train_val_indices[0]]],
				'validation_data': ([train_data[0][train_val_indices[1]], train_data[1][train_val_indices[1]]], train_labels[train_val_indices[1]])
			})
		else:
			args.update({
				'x': train_data[train_val_indices[0]],
				'validation_data': (train_data[train_val_indices[1]], train_labels[train_val_indices[1]])
			})
		args['y'] = train_labels[train_val_indices[0]]

	history = model.fit(**args)

	# if callback is None:
	# 	if validationRatio == 0:
	# 		history = model.fit(train_data,train_labels, epochs = nEpoch, steps_per_epoch = batchSize, verbose = verbose)
	# 	else:
	# 		history = model.fit(train_data,train_labels, epochs = nEpoch, validation_split = validationRatio,
	# 							steps_per_epoch = batchSize, verbose = verbose, validation_steps = batchSize)
	# else:
	# 	if validationRatio == 0:
	# 		history = model.fit(train_data,train_labels, epochs = nEpoch, steps_per_epoch = batchSize, verbose = verbose, callbacks = callback)
	# 	else:
	# 		history = model.fit(train_data,train_labels, epochs = nEpoch, validation_split = validationRatio, steps_per_epoch = batchSize,
	# 							verbose = verbose, validation_steps = batchSize, callbacks = callback)

	return history


def saveBetas(fileInputName, model, filePath = '', saveExtension = '', verbose = True):
	betas_layer = model.get_layer(name = 'Utilities')
	betas = betas_layer.get_weights()
	if verbose:
		print(betas)
	np.save(filePath + 'Betas_'+ fileInputName + '_' + saveExtension +'.npy', betas)
	return betas

def normalize(data):
	return (data-data.mean(axis=0))/(data.std(axis=0))
