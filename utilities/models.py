import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, Add, Reshape


""" Definition of Keras Models : 
			- MNL
			- dense NN (simple + extra input)
			- L-MNL (simple + extra input)
			- Hruschka04
			- Hruschka07
"""

def MNL(beta_num, choices_num, minima = None, logits_activation = 'softmax'):
	""" Multinomial Logit model as a CNN, Linear-in-parameters """

	main_input= Input((beta_num, choices_num,1), name = 'Features')
	if minima is None:
		utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias=False, trainable=True)(main_input)
	else:
		utilities = Conv2D(filters=1, kernel_size = [beta_num,1], strides=(1,1), padding ='valid', name='Utilities',
			use_bias=False, weights=[minima], trainable=True)(main_input)

	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)
	logits = Activation(logits_activation, name='Choice')(utilitiesR)
	model = Model(inputs=main_input, outputs=logits, name='Choice')
	return model



def denseNN(beta_num, choices_num, networkSize = 16, logits_activation = 'softmax'):
	""" Dense Neural Network (writen with CNN for coding convenience. Connections are Equivalent Here)
	 	- Kernel is size of Input. Numbers of Filters are the size of the neurons ( == DNN )
	"""
	main_input= Input((beta_num, choices_num,1), name='Features')
	dense = Conv2D(filters = networkSize, kernel_size=[beta_num, choices_num], activation = 'relu', padding = 'valid', name = 'Dense_NN_per_frame')(main_input)
	#Dropout successfully prevents overfit. If removed, it is better to run model.fit on full data including a callback.
	dropped = Dropout(0.2, name = 'Regularizer')(dense)
	new_feature = Dense(units = choices_num, name="Output_new_feature")(dropped)
	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)

	logits = Activation(logits_activation, name='Choice')(new_featureR)

	model = Model(inputs = main_input, outputs=logits)
	return model

def denseNN_extra(beta_num, choices_num, nExtraFeatures, networkSize = 16, logits_activation = 'softmax'):
	""" Dense Neural Network, using the second inputs given as only input """
	main_input= Input((beta_num, choices_num,1), name='Features')
	extra_input = Input((nExtraFeatures,1,1), name='Extra_Input')

	dense = Conv2D(filters = networkSize, kernel_size=[nExtraFeatures, 1], activation='relu', padding='valid', name = 'Dense_NN_per_frame')(extra_input)
	dropped = Dropout(0.2, name='Regularizer')(dense)
	new_feature = Dense(units=choices_num, name="Output_new_feature")(dropped)

	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)

	logits = Activation(logits_activation, name='Choice')(new_featureR)

	model = Model(inputs = extra_input, outputs=logits)
	return model



def enhancedMNL_sameInput(beta_num, choices_num, networkSize = 16, train_betas = True, minima = None, hidden_layers = 1, logits_activation = 'softmax'):
	""" L-MNL Model with violation of separated input constraint
		1) Main Input (X) follows the MNL architecture till Utilities
		2) Main Input (X=Q) follows the DNN architecture till new feature
		3) Both terms are added to make Final Utilities
	"""
	main_input = Input((beta_num, choices_num,1), name='Features')

	# Standard MNL from DCM:
	if minima is None:
		utilities = utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias=False, trainable=train_betas)(main_input)
	else:
		utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias=False, weights=[minima], trainable=train_betas)(main_input)

	# Dense Neural Network
	dense = Conv2D(filters = networkSize, kernel_size=[beta_num, choices_num], activation='relu', padding='valid', name='Dense_NN_per_frame')(main_input)
	dropped = Dropout(0.2, name='Regularizer')(dense)
	new_feature = Dense(units=choices_num, name="Output_new_feature")(dropped)

	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)
	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)

	#Enhancing utility functions with dense NN:
	final_utilities = Add(name="New_Utility_functions")([utilitiesR, new_featureR])
	logits = Activation(logits_activation, name='Choice')(final_utilities)

	model = Model(inputs=main_input, outputs=logits)
	return model


# Mixed model with 2 inputs (input for beta and choices for MNL, input of extra features for denseNN) 
def enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize, hidden_layers=1, train_betas=True,
						   minima=None, logits_activation='softmax'):
	""" L-MNL Model
		1) Main Input (X) follows the MNL architecture till Utilities
		2) Extra Input (Q) follows the DNN architecture till new feature
		3) Both terms are added to make Final Utilities
	"""

	main_input = Input((beta_num, choices_num,1), name='Features')
	extra_input = Input((nExtraFeatures,1,1), name='Extra_Input')

	if minima is None:
		utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias=False, trainable=train_betas)(main_input)
	else:
		utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias = False, weights=minima, trainable=train_betas)(main_input)

	dense = Conv2D(filters=networkSize, kernel_size=[nExtraFeatures, 1], activation='relu', padding='valid', name='Dense_NN_per_frame')(extra_input)
	dropped = Dropout(0.2, name='Regularizer')(dense)
	
	x = dropped
	for i in range(hidden_layers-1):
		x = Dense(units=networkSize, activation='relu', name="Dense{}".format(i))(x)
		x = Dropout(0.2, name='Drop{}'.format(i))(x)
	dropped = x

	new_feature = Dense(units=choices_num, name="Output_new_feature")(dropped)
	#new_feature = Dense(units = choices_num, name = "Output_new_feature")(dense)
	
	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)
	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)

	final_utilities = Add(name="New_Utility_functions")([utilitiesR, new_featureR])


	logits = Activation(logits_activation, name='Choice')(final_utilities)

	model = Model(inputs=[main_input,extra_input], outputs=logits)
	return model



def Hruschka_multi(beta_num, choices_num, networkSize = 100, logits_activation = 'softmax'):
	""" Shared DNN Network
			- Main input separated by alternatives, then fed to same DNN layer and concatenated
	"""
	main_input = []
	inputs = []
	dense = []
	dropout = []
	new_feature = []
	new_featureR = []
	dense_layer = Conv2D(filters=networkSize, kernel_size=[beta_num, 1], activation='relu',
						  padding = 'valid', name = 'Dense_NN_per_frame')
	for i in range(choices_num):
		main_input.append(Input((beta_num, 1, 1), name='Features_{}'.format(i)))
		inputs.append( Reshape(target_shape=(beta_num,1,1))(main_input[i]))
		dense.append(dense_layer(inputs[i]))
		# Fully connected to image size beta_num X choices_num
		dropout.append( Dropout(0.2, name='Regularizer_{}'.format(i))(dense[i]))
		new_feature.append( Dense(units=1, name="Output_new_feature_{}".format(i))(dropout[i]))
		new_featureR.append( Reshape([1], name='Remove_Dim_{}'.format(i))(new_feature[i]))

	concatenated = Concatenate()(new_featureR)
	logits = Activation(logits_activation, name='Choice')(concatenated)

	model = Model(inputs=main_input, outputs=logits)
	return model



def Hruschka_multi07(beta_num, choices_num, networkSize = 100, logits_activation = 'softmax'):
	""" Shared DNN Network + MNL
			- Main input follows the Architecture of MNL till Utilities
			- Main input separated by alternatives, then fed to same DNN layer and concatenated
			- Both terms added for new utility functions
	"""
	main_input = []
	inputs = []
	dense = []
	dropout = []
	new_feature = []
	new_featureR = []

	#ANN component with shared weigths
	dense_layer = Conv2D(filters=networkSize, kernel_size=[beta_num, 1], activation='relu',
						  padding='valid', name='Dense_NN_per_frame')
	for i in range(choices_num):
		main_input.append(Input((beta_num, 1, 1), name='Features_{}'.format(i)))
		inputs.append( Reshape(target_shape=(beta_num,1,1))(main_input[i]))
		dense.append(dense_layer(inputs[i]))
		# Fully connected to image size beta_num X choices_num
		dropout.append( Dropout(0.2, name='Regularizer_{}'.format(i))(dense[i]) )
		new_feature.append( Dense(units=1, name="Output_new_feature_{}".format(i))(dropout[i]) )
		new_featureR.append( Reshape([1], name='Remove_Dim_{}'.format(i))(new_feature[i]) )

	concatenated = Concatenate()(new_featureR)

	#Linear component
	concatenated_input = Concatenate(axis=2)(main_input)
	utilities = Conv2D(filters=1, kernel_size=[beta_num, 1], strides=(1, 1), padding='valid', name='Utilities',
					   use_bias=False)(concatenated_input)
	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)

	# Outputs and loss
	final_utilities = Add(name="New_Utility_functions")([utilitiesR, concatenated])

	logits = Activation(logits_activation, name='Choice')(final_utilities)

	model = Model(inputs=main_input, outputs=logits)
	return model



# https://github.com/keras-team/keras/issues/5920
# An optimizer that learns selected layers at a different rate.
# Need to add in source code if used. Follow instructions 
def multiAdamOptimizer(model, layerList, multiplier = 0.8):
	beta_layer_variables = list()
	for layer in model.layers:
	    if layer.name in layerList:
	        beta_layer_variables.extend(layer.weights)
	multiadam = multiAdam(multiplied_vars=beta_layer_variables, multiplier = multiplier)
	
	return multiadam



