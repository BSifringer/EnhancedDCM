from . import models as mdl
from . import train_utils as tu
from keras.optimizers import RMSprop, Adam, SGD 

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


def runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart = '',
		   saveName = '', loss='categorical_crossentropy', logits_activation = 'softmax'):
	saveExtension = 'MNL' + saveName

	model = mdl.MNL(beta_num, choices_num, logits_activation=logits_activation)
	optimizer = Adam(clipnorm = 50.)
	model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)

	tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
					   saveExtension = saveExtension, filePart = filePart )

	betas = tu.saveBetas(fileInputName, model, filePath = filePath, saveExtension = saveExtension)


	#plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')

	return betas, saveExtension


def runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		  extraInput = False, nExtraFeatures = None, filePart = '', saveName = '', networkSize=100,
		  loss='categorical_crossentropy', logits_activation = 'softmax'):

	saveExtension = 'NN' + saveName
	if extraInput:
		saveExtension = saveExtension + 'extra'
		model = mdl.denseNN_extra(beta_num, choices_num, nExtraFeatures, networkSize = networkSize, logits_activation=logits_activation)
	else:
		model = mdl.denseNN(beta_num, choices_num, networkSize=networkSize, logits_activation=logits_activation)

	optimizer = Adam(clipnorm = 50.)
	model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)
	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
							saveExtension = saveExtension, filePart = filePart, NN = True)
	else:
		tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
						   saveExtension = saveExtension, filePart = filePart)

	#plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')
	return saveExtension


def runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		extraInput = False, nExtraFeatures = None, minima = None, train_betas = True, filePart = '', saveName = '',
		networkSize = 100, verbose = 0, hidden_layers=1, loss='categorical_crossentropy', logits_activation = 'softmax'):

	saveExtension = 'Enhanced' + saveName
	if extraInput:
		saveExtension = saveExtension + 'extra'
		model = mdl.enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize = networkSize, minima = minima,
										   train_betas = train_betas, hidden_layers=hidden_layers, logits_activation=logits_activation)
	else:
		model = mdl.enhancedMNL_sameInput(beta_num, choices_num, minima = minima, train_betas = train_betas,
										  hidden_layers=hidden_layers, logits_activation=logits_activation)

	#optimizer = SGD(momentum = 0.2, decay = 0.001)
	optimizer = Adam(clipnorm=50.)
	model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)

	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
							saveExtension = saveExtension, filePart = filePart, verbose = verbose )
	else:
		tu.train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath = filePath,
						   saveExtension = saveExtension, filePart = filePart, verbose = verbose )

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

