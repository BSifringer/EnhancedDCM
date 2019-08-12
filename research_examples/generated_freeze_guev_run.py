
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-1])
    path.append(dir(parent))

import utilities.run_utils as ru
import utilities.models as mdl
import utilities.train_utils as tu
import numpy as np
# Need to code thes imports better!
from generated_guev import data_manager as generatedDM
from keras.utils import np_utils, plot_model
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD
import tensorflow as tf

"""
	This script is specific to a single experiment to compare joint optimization versus sequential optimization of both model components.
	The Method consists in freezing layers during training time.
	This script should not be taken as an example for easily training a model. Refer to generated_run.py or semi_synthetic_run.py instead.
"""

simpleArchitecture = True

if simpleArchitecture:
	beta_num = 3
	nExtraFeatures = 3
else:

	beta_num = 4
	nExtraFeatures = 3

choices_num = 2
batchSize = 50


def runfreezeNNMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		extraInput = False, nExtraFeatures = None, minima = None, train_betas = True, filePart = '', saveName = '',
		networkSize = 100, loss='categorical_crossentropy', logits_activation = 'softmax'):
	""" This method trains the NN first, then the Betas """

	saveExtension = 'freezeNN' + saveName

	main_input= Input((beta_num, choices_num,1), name = 'Features')
	extra_input = Input((nExtraFeatures,1,1), name = 'Extra_Input')

	if minima is None:
		utilities = Conv2D(filters=1, kernel_size = [beta_num,1], strides=(1,1), padding ='valid', name = 'Utilities',
			use_bias = False, trainable = False)(main_input)
	else:
		utilities = Conv2D(filters=1, kernel_size = [beta_num,1], strides=(1,1), padding ='valid', name = 'Utilities',
			use_bias = False, weights = [minima], trainable = False)(main_input)

	dense = Conv2D(filters = networkSize, kernel_size = [nExtraFeatures, 1], activation = 'relu', padding = 'valid', name = 'Dense_NN_per_frame')(extra_input)
	dropped = Dropout(0.2, name = 'Regularizer')(dense)
	new_feature = Dense(units = choices_num, name = "Output_new_feature")(dropped)
	#new_feature = Dense(units = choices_num, name = "Output_new_feature")(dense)

	new_featureR = Reshape([choices_num], name = 'Remove_Dim')(new_feature)
	utilitiesR = Reshape([choices_num], name = 'Flatten_Dim')(utilities)

	final_utilities = Add(name = "New_Utility_functions")([utilitiesR, new_featureR])


	logits = Activation(logits_activation, name='Choice')(final_utilities)

	frozen_model = Model(inputs = [main_input,extra_input], outputs = logits)


	utilities.trainable = False
	optimizer = Adam()
	frozen_model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)

	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, frozen_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart, verbose = 1)
	else:
		tu.train_sameInput(fileInputName, nEpoch, frozen_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart)


	print("Untrained Betas")
	betas = tu.saveBetas(fileInputName, frozen_model, filePath = filePath, saveExtension = saveExtension)
	plot_model(frozen_model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')

	dense = frozen_model.get_layer('Dense_NN_per_frame')
	new_feature = frozen_model.get_layer('Output_new_feature')
	utilities = frozen_model.get_layer('Utilities')

	dense.trainable = False,
	new_feature.trainable = False,
	utilities.trainable = False,

	betas = utilities.get_weights()
	dense_weigths = dense.get_weights()
	feature_weigths = new_feature.get_weights()


	main_input2= Input((beta_num, choices_num,1), name = 'Features')
	extra_input2 = Input((nExtraFeatures,1,1), name = 'Extra_Input')



	utilities2 = Conv2D(filters=1, kernel_size = [beta_num,1], strides=(1,1), padding ='valid', name = 'Utilities',
			use_bias = False, weights = betas, trainable = True)(main_input2)

	dense2 = Conv2D(filters = networkSize,trainable = False, weights = dense_weigths,kernel_size = [nExtraFeatures, 1], activation = 'relu', padding = 'valid', name = 'Dense_NN_per_frame')(extra_input2)
	dropped2 = Dropout(0.2, name = 'Regularizer')(dense2)
	new_feature2 = Dense(units = choices_num, trainable = False, weights = feature_weigths, name = "Output_new_feature")(dropped2)
	#new_feature = Dense(units = choices_num, name = "Output_new_feature")(dense)

	new_featureR2 = Reshape([choices_num], name = 'Remove_Dim')(new_feature2)
	utilitiesR2 = Reshape([choices_num], name = 'Flatten_Dim')(utilities2)

	final_utilities2 = Add(name = "New_Utility_functions")([utilitiesR2, new_featureR2])


	logits2 = Activation(logits_activation, name='Choice')(final_utilities2)

	trainable_model = Model(inputs = [main_input2,extra_input2], outputs = logits2)
	trainable_model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)

	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, trainable_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart, verbose = 1)
	else:
		tu.train_sameInput(fileInputName, nEpoch, trainable_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart)

	print("Trained Betas")
	betas = tu.saveBetas(fileInputName, trainable_model, filePath = filePath, saveExtension = saveExtension)
	plot_model(trainable_model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')

	return betas, saveExtension



def runfreezeBetaMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
		extraInput = False, nExtraFeatures = None, minima = None, train_betas = True, filePart = '', saveName = '',
		networkSize = 100, loss='categorical_crossentropy', logits_activation = 'softmax'):
	""" This method trains the betas first, then the NN """


	saveExtension = 'freezeB' + saveName

	main_input= Input((beta_num, choices_num,1), name = 'Features')
	extra_input = Input((nExtraFeatures,1,1), name = 'Extra_Input')

	if minima is None:
		utilities = Conv2D(filters=1, kernel_size = [beta_num,1], strides=(1,1), padding ='valid', name = 'Utilities',
			use_bias = False, trainable = train_betas)(main_input)
	else:
		utilities = Conv2D(filters=1, kernel_size = [beta_num,1], strides=(1,1), padding ='valid', name = 'Utilities',
			use_bias = False, weights = [minima], trainable = train_betas)(main_input)

	dense = Conv2D(filters = networkSize, trainable = False, kernel_size = [nExtraFeatures, 1], activation = 'relu', padding = 'valid', name = 'Dense_NN_per_frame')(extra_input)
	dropped = Dropout(0.2, name = 'Regularizer')(dense)
	new_feature = Dense(units = choices_num, trainable = False, name = "Output_new_feature")(dropped)
	#new_feature = Dense(units = choices_num, name = "Output_new_feature")(dense)

	new_featureR = Reshape([choices_num], name = 'Remove_Dim')(new_feature)
	utilitiesR = Reshape([choices_num], name = 'Flatten_Dim')(utilities)

	final_utilities = Add(name = "New_Utility_functions")([utilitiesR, new_featureR])


	logits = Activation(logits_activation, name='Choice')(final_utilities)

	frozen_model = Model(inputs = [main_input,extra_input], outputs = logits)


	dense.trainable = False
	new_feature.trainable = False
	optimizer = Adam(clipnorm = 50.)
	frozen_model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)

	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, frozen_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart)
	else:
		tu.train_sameInput(fileInputName, nEpoch, frozen_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart)

	print("Trained only Betas")
	betas = tu.saveBetas(fileInputName, frozen_model, filePath = filePath, saveExtension = saveExtension)
	plot_model(frozen_model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')




	utilities.trainable = False
	dense.trainable = True
	new_feature.trainable = True

	trainable_model = Model(inputs = [main_input,extra_input], outputs = logits)

	optimizer = Adam(clipnorm = 50.)
	trainable_model.compile(optimizer = optimizer, metrics = ["accuracy"], loss = loss)

	if extraInput:
		tu.train_extraInput(fileInputName, nEpoch, trainable_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart)
	else:
		tu.train_sameInput(fileInputName, nEpoch, trainable_model, train_data_name, batchSize, filePath = filePath, saveExtension = saveExtension, filePart = filePart)
	print("Same Betas")
	betas = tu.saveBetas(fileInputName, trainable_model, filePath = filePath, saveExtension = saveExtension)
	plot_model(trainable_model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')

	return betas, saveExtension



def GeneratedMNL(filePath, fileInputName, beta_num, choices_num, train_data_name,
	filePart = '', saveName = '', loss='categorical_crossentropy', logits_activation = 'softmax'):

	nEpoch = 150

	return ru.runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart, saveName = saveName, loss=loss, logits_activation=logits_activation)


def GeneratedNN(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name,
	extraInput = False, filePart = '',  saveName = '', loss='categorical_crossentropy', logits_activation = 'softmax'):

	nEpoch = 400

	return ru.runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name,  batchSize, extraInput, nExtraFeatures, filePart, saveName = saveName, loss=loss, logits_activation=logits_activation)


def GeneratedMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures,
	train_data_name, extraInput = False, minima = None, train_betas = True, filePart = '',
	saveName = '', networkSize = 16, verbose = 0, loss='categorical_crossentropy', logits_activation = 'softmax'):

	nEpoch = 200

	return ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput, nExtraFeatures, minima, train_betas, filePart, saveName = saveName, networkSize = networkSize, verbose = verbose, loss=loss, logits_activation=logits_activation)


if __name__ == '__main__':


	filePath = 'generated_guev/freeze/'
	extensions = ['_train', '_test']

	lmnlArchitecture = True
	beta_num = 2
	nExtraFeatures = 3
	choices_num = 2
	fileInputName = 'generated'
	_,_,train_data_name, beta_num, nExtraFeatures = generatedDM.keras_input(filePath,fileInputName, filePart = extensions[0], lmnlArchitecture = lmnlArchitecture) # create keras input for train set
	list = [100]
	for i in list:
		print("{} Neurons -------".format(i))
		nEpoch = 150
		extraInput = True
		print("Betas Freeze Model")
		with tf.variable_scope("Betas_"):
			_, saveExtension = runfreezeBetaMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput, nExtraFeatures, None, True, filePart = extensions[0], networkSize = i, saveName = "BetaFreeze")

		betas, saveExtension = GeneratedMNL(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart = extensions[0], saveName = "BetaFreeze")
		_,saveExtension = GeneratedMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = extraInput, minima = np.array(betas), train_betas = False, filePart=extensions[0], saveName = "BetaFreeze", networkSize = i)

		print("NN Freeze Model")
		#betas,saveExtension = GeneratedMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, extraInput = extraInput, train_betas = False, filePart=extensions[0], saveName = "NNFreeze", networkSize = i, verbose = 1)
		with tf.variable_scope("NN_"):
			_, saveExtension = runfreezeNNMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput, nExtraFeatures, None, True, filePart = extensions[0], networkSize = i, saveName = "NNFreeze2")
#			generatedDM.biogeme_input(choices_num, filePath, fileInputName, saveExtension, extraInput = True, extensions = extensions, simpleArchitecture = simpleArchitecture)

		_,saveExtension = GeneratedMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = extraInput, filePart=extensions[0], networkSize = i, verbose = 1)
