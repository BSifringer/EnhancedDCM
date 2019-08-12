from . import models as mdl
from . import train_utils as tu
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt

""" Utilities to compile a model (loss, optimizer) and sends to training

	Possible Models:
	- MNL
	- NN (DNN)
	- Mixed (L-MNL)
	- Hruschka (shared DNN)
	- Hruschka2007 (Hybrid, shared DNN)
"""

"""
	When Applicable

:param filePath: 		Where to save the trained model
:param fileInputName: 	Name of Trained model
:param beta_num: 		Number of inputs in X
:param choices_num: 	Number of Alternatives (Output dimension)
:param nEpoch: 			Training Cycles
:param train_data_name: Name of .npy with inputs
:param batchSize: 		Training batch size
:param extraInput: 		If model has a Q input set
:param nExtraFeatures: 	Size of input to NN component
:param minima: 			Initialize CNN kernel (Betas) with minima
:param train_betas: 	Set to False to fix kernel of CNN
:param filePart: 		Dataset type (e.g. _train)
:param saveName: 		SaveName extension for model differentiation
:param networkSize: 	Number of Neurons in DNN layers
:param verbose: 		Keras verbose of model.fit  0 = silent, 1= batch and instant metrics, 2= Epochs with metrics
:param hidden_layers: 	Number of DNN layers (only implemented for L-MNL)
:param loss: 			Loss function
:param logits_activation: Activation of output layer

:return: - betas: 			beta values (CNN kernel)
		 - saveExtension: 	model's full extension name
"""


def runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart = '', betas_save = True, train_val_indices = None,
		   saveName = '', loss='categorical_crossentropy', logits_activation = 'softmax', verbose = 0, validationRatio = 0, callback = None, model_inputs = None):
	saveExtension = 'MNL' + saveName

	model = mdl.MNL(beta_num, choices_num, logits_activation=logits_activation)
	optimizer = Adam(clipnorm = 50.)
	model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)

	tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
					   saveExtension = saveExtension, filePart = filePart, verbose = verbose, validationRatio = validationRatio,
					   callback = callback, model_inputs = model_inputs, train_val_indices = train_val_indices)
	betas = None
	if betas_save:
		betas = tu.saveBetas(fileInputName, model, filePath = filePath, saveExtension = saveExtension)


	#plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')

	return betas, [saveExtension, model]


def runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		  extraInput = False, nExtraFeatures = None, filePart = '', saveName = '', networkSize=100, dropout=0.2, regularizer=None, hidden_layers=1,  e_normalize=False,
		  loss='categorical_crossentropy', logits_activation = 'softmax', verbose = 0, validationRatio = 0, callback = None, model_inputs = None):

	saveExtension = 'NN' + saveName
	if extraInput:
		saveExtension = saveExtension + 'extra'
		model = mdl.denseNN_extra(beta_num, choices_num, nExtraFeatures, networkSize = networkSize, logits_activation=logits_activation,
				dropout=dropout, regularizer=regularizer, hidden_layers=hidden_layers)
	else:
		model = mdl.denseNN(beta_num, choices_num, networkSize=networkSize, logits_activation=logits_activation,
				dropout=dropout, regularizer=regularizer, hidden_layers=hidden_layers)

	optimizer = Adam(clipnorm = 50.)
	model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)
	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
							saveExtension = saveExtension, filePart = filePart, NN = True, verbose = verbose, validationRatio = validationRatio,
							callback = callback, model_inputs = model_inputs,  e_normalize=e_normalize)
	else:
		tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
						   saveExtension = saveExtension, filePart = filePart, verbose = verbose, validationRatio = validationRatio,
						   callback = callback, model_inputs = model_inputs)

	#plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')
	return saveExtension


def runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		extraInput = False, nExtraFeatures = None, minima = None, train_betas = True, filePart = '',
		saveName = '', networkSize = 100, betas_save = True, train_val_indices = None, e_normalize=False,
		verbose = 0, hidden_layers=1, loss='categorical_crossentropy', logits_activation = 'softmax',
		validationRatio = 0, callback = None, model_inputs = None, dropout = 0.2, regularizer = None, plot=True):

	saveExtension = 'Enhanced' + saveName

	if extraInput:
		saveExtension = saveExtension + 'extra'
		model = mdl.enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize = networkSize,
				minima = minima, train_betas = train_betas, hidden_layers=hidden_layers, logits_activation=logits_activation,
				dropout = dropout, regularizer = regularizer)
	else:
		model = mdl.enhancedMNL_sameInput(beta_num, choices_num, minima = minima, train_betas = train_betas,
										  hidden_layers=hidden_layers, logits_activation=logits_activation)

	#optimizer = SGD(momentum = 0.2, decay = 0.001)
	optimizer = Adam(clipnorm=50.)
	optimizer = Adam()
	model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)

	if extraInput:
		history = tu.train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart,  e_normalize=e_normalize,
			verbose = verbose, validationRatio = validationRatio, callback = callback, model_inputs = model_inputs, train_val_indices = train_val_indices)
	else:
		tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
						   saveExtension = saveExtension, filePart = filePart, verbose = verbose, validationRatio = validationRatio,
						   callback = callback, model_inputs = model_inputs , train_val_indices = train_val_indices)
	betas = None
	if betas_save:
		betas = tu.saveBetas(fileInputName, model, filePath = filePath, saveExtension = saveExtension)


	if plot:
		# summarize history for accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()


	#plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')
	return betas, saveExtension

def runNested(filePath, fileInputName, beta_num, nested_dict, nEpoch, train_data_name, batchSize,
		extraInput = False, nExtraFeatures = None, minima = None, train_betas = True, filePart = '', saveName = '', networkSize = 100, betas_save = True, train_val_indices = None, e_normalize=False,
		verbose = 0, hidden_layers=1, loss='categorical_crossentropy', logits_activation = 'softmax', validationRatio = 0, callback = None, model_inputs = None, dropout = 0.2, regularizer = None, save_weights = True):

	saveExtension = 'Nest' + saveName

	model = mdl.L_Nested(beta_num, nested_dict, nExtraFeatures, networkSize = networkSize, minima = minima, train_betas = train_betas, hidden_layers=hidden_layers, logits_activation=logits_activation, dropout = dropout, regularizer = regularizer)

	#optimizer = SGD(momentum = 0.2, decay = 0.001)
	optimizer = Adam(clipnorm=50.)
	optimizer = Adam()
	model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)

	tu.train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart, save_weights=save_weights,
			verbose = verbose, validationRatio = validationRatio, callback = callback, model_inputs = model_inputs, train_val_indices = train_val_indices, e_normalize=e_normalize)

	betas = None
	if betas_save:
		betas = tu.saveBetas(fileInputName, model, filePath = filePath, saveExtension = saveExtension)

	plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')
	for i in range(len(nested_dict.keys())):
		print(model.get_layer('nest_value_{}'.format(i)).get_weights())
	return model

def runSub(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		extraInput = False, nExtraFeatures = None, minima = None, train_betas = True, filePart = '', saveName = '', networkSize = 100, betas_save = True, train_val_indices = None,
		verbose = 0, hidden_layers=1, loss='categorical_crossentropy', logits_activation = 'softmax', validationRatio = 0, callback = None, model_inputs = None, dropout = 0.2, regularizer = None):

	saveExtension = 'Sub' + saveName

	model = mdl.enhancedMNL_subnets(beta_num, choices_num, nExtraFeatures, networkSize = networkSize, minima = minima, train_betas = train_betas, hidden_layers=hidden_layers, logits_activation=logits_activation, dropout = dropout, regularizer = regularizer)

	#optimizer = SGD(momentum = 0.2, decay = 0.001)
	optimizer = Adam(clipnorm=50.)
	optimizer = Adam()
	model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)

	tu.train_subInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart,
			verbose = verbose, validationRatio = validationRatio, callback = callback, model_inputs = model_inputs, train_val_indices = train_val_indices)

	betas = None
	if betas_save:
		betas = tu.saveBetas(fileInputName, model, filePath = filePath, saveExtension = saveExtension)

	#plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')
	return betas, saveExtension


def runHrusch(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart = '',
			  saveName = '', loss='categorical_crossentropy', logits_activation = 'softmax', networkSize=100):

	saveExtension = 'Hruschka' + saveName

	model = mdl.Hruschka_multi(beta_num, choices_num, networkSize=networkSize, logits_activation=logits_activation)
	optimizer = Adam(clipnorm=50.)
	model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)
	Hrusch = True
	tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
					   saveExtension=saveExtension, filePart=filePart, Hrusch=Hrusch )

	return saveExtension

def runHrusch07(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart = '',
			  saveName = '', loss='categorical_crossentropy', logits_activation = 'softmax', networkSize=100):

	saveExtension = 'Hruschka07' + saveName

	model = mdl.Hruschka_multi07(beta_num, choices_num, networkSize=networkSize, logits_activation=logits_activation)
	optimizer = Adam(clipnorm=50.)
	model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)
	Hrusch = True
	tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
					   saveExtension=saveExtension, filePart=filePart, Hrusch=Hrusch )
	return saveExtension
