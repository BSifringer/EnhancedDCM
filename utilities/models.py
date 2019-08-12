import numpy as np
from keras.models import Model
from keras.initializers import Constant
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, Add, Reshape, Layer, Lambda, Multiply, Dot
from keras.regularizers import l2
from keras import backend as K
from keras.constraints import MinMaxNorm
from keras import constraints

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



def denseNN(beta_num, choices_num, networkSize = 16, logits_activation = 'softmax', dropout=0.2, regularizer=None, hidden_layers=1):
	""" Dense Neural Network (writen with CNN for coding convenience. Connections are Equivalent Here)
	 	- Kernel is size of Input. Numbers of Filters are the size of the neurons ( == DNN )
	"""
	main_input= Input((beta_num, choices_num,1), name='Features')
	dense = Conv2D(filters = networkSize, kernel_size=[beta_num, choices_num], activation = 'relu', padding = 'valid', name = 'Dense_NN_per_frame',kernel_regularizer=regularizer)(main_input)
	#Dropout successfully prevents overfit. If removed, it is better to run model.fit on full data including a callback.
	dropped = Dropout(dropout, name = 'Regularizer')(dense)
	x = dropped
	for i in range(hidden_layers-1):
		x = Dense(units=networkSize, activation='relu', name="Dense{}".format(i))(x)
		x = Dropout(dropout, name='Drop{}'.format(i))(x)
	dropped = x

	new_feature = Dense(units=choices_num, name="Output_new_feature")(dropped)
	new_feature = Dense(units = choices_num, name="Output_new_feature")(dropped)
	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)

	logits = Activation(logits_activation, name='Choice')(new_featureR)

	model = Model(inputs = main_input, outputs=logits)
	return model

def denseNN_extra(beta_num, choices_num, nExtraFeatures, networkSize = 16, logits_activation = 'softmax', dropout=0.2, regularizer=None, hidden_layers=1):
	""" Dense Neural Network, using the second inputs given as only input """
	main_input= Input((beta_num, choices_num,1), name='Features')
	extra_input = Input((nExtraFeatures,1,1), name='Extra_Input')

	dense = Conv2D(filters = networkSize, kernel_size=[nExtraFeatures, 1], activation='relu', padding='valid', name = 'Dense_NN_per_frame',  kernel_regularizer=regularizer)(extra_input)
	dropped = Dropout(dropout, name='Regularizer')(dense)
	x = dropped
	for i in range(hidden_layers-1):
		x = Dense(units=networkSize, activation='relu', name="Dense{}".format(i))(x)
		x = Dropout(dropout, name='Drop{}'.format(i))(x)
	dropped = x
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
def enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize, hidden_layers=1, train_betas=True, minima=None, logits_activation='softmax', dropout = 0.2, regularizer = None):
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

	dense = Conv2D(filters=networkSize, kernel_size=[nExtraFeatures, 1], activation='relu', padding='valid', name='Dense_NN_per_frame', kernel_regularizer=regularizer)(extra_input)
	dropped = Dropout(dropout, name='Regularizer')(dense)

	x = dropped
	for i in range(hidden_layers-1):
		x = Dense(units=networkSize, activation='relu', name="Dense{}".format(i))(x)
		x = Dropout(dropout, name='Drop{}'.format(i))(x)
	dropped = x

	new_feature = Dense(units=choices_num, name="Output_new_feature")(dropped)
	#new_feature = Dense(units = choices_num, name = "Output_new_feature")(dense)

	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)
	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)

	final_utilities = Add(name="New_Utility_functions")([utilitiesR, new_featureR])


	logits = Activation(logits_activation, name='Choice')(final_utilities)

	model = Model(inputs=[main_input,extra_input], outputs=logits)
	return model

# Mixed model with 2 inputs (input for beta and choices for MNL, input of extra features for denseNN)
def enhancedMNL_subnets(beta_num, choices_num, nExtraFeatures, networkSize, hidden_layers=1, train_betas=True, minima=None, logits_activation='softmax', dropout = 0.2, regularizer = None):
	""" L-MNL Model
		1) Main Input (X) follows the MNL architecture till Utilities
		2) Extra Input (Q) follows the DNN architecture till new feature
		2.5) (Q) is separated per choice number
		3) Both terms are added to make Final Utilities
	"""

	main_input = []
	inputs = []
	dense = []
	dropout = []
	new_feature = []
	new_featureR = []

	choice_input = Input((beta_num, choices_num,1), name='Features')
	main_input.append(choice_input)

	#ANN component with shared weigths
	# dense_layer = Conv2D(filters=networkSize, kernel_size=[beta_num, 1], activation='relu',
	# 					  padding='valid', name='Dense_NN_per_frame')
	for i in range(choices_num):
		main_input.append(Input((nExtraFeatures, 1, 1), name='Features_{}'.format(i)))
		inputs.append(Reshape(target_shape=(nExtraFeatures,1,1))(main_input[i+1]))
		# dense.append(dense_layer(inputs[i]))
		dense.append(Conv2D(filters=networkSize, kernel_size=[nExtraFeatures, 1], activation='relu',
						  padding='valid', name='Dense_NN_per_frame_{}'.format(i))(inputs[i]))
		# Fully connected to image size beta_num X choices_num
		dropout.append(Dropout(0.2, name='Regularizer_{}'.format(i))(dense[i]) )
		new_feature.append( Dense(units=1, name="Output_new_feature_{}".format(i))(dropout[i]) )
		new_featureR.append( Reshape([1], name='Remove_Dim_{}'.format(i))(new_feature[i]) )

	if choices_num>1:
		concatenated = Concatenate()(new_featureR)
	else:
		concatenated = new_featureR[0]
	#Linear component
#	concatenated_input = Concatenate(axis=2)(main_input)
	utilities = Conv2D(filters=1, kernel_size=[beta_num, 1], strides=(1, 1), padding='valid', name='Utilities',
					   use_bias=False)(choice_input)
	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)

	# Outputs and loss
	final_utilities = Add(name="New_Utility_functions")([utilitiesR, concatenated])

	logits = Activation(logits_activation, name='Choice')(final_utilities)

	model = Model(inputs=main_input, outputs=logits)
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

from keras import backend as K
from keras.layers import Layer

class nestMultiply(Layer):

	def __init__(self, W_constraint=None, **kwargs):
		self.W_constraint = constraints.get(W_constraint)
		super(nestMultiply, self).__init__(**kwargs)

	def build(self, input_shape):
			# Create a trainable weight variable for this layer.
		self.W = self.add_weight(name='W',
                          			  shape=([1]),
                                      initializer='ones',
									  dtype='float32',
                                      trainable=True,
									  constraint = self.W_constraint)
		super(nestMultiply, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		return K.tf.multiply(x, self.W)

	def compute_output_shape(self, input_shape):
		return input_shape


class nestDivide(Layer):

	def __init__(self, W_constraint=None, **kwargs):
		self.W_constraint = constraints.get(W_constraint)
		super(nestDivide, self).__init__(**kwargs)

	def build(self, input_shape):
			# Create a trainable weight variable for this layer.
		self.W = self.add_weight(name='W',
                          			  shape=([1]),
                                      initializer='ones',
									  dtype='float32',
                                      trainable=True,
									  constraint = self.W_constraint)
		super(nestDivide, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		return K.tf.math.divide(x, self.W)

	def compute_output_shape(self, input_shape):
		return input_shape

def create_shared_weights(conv1, conv2, input_shape):
    with K.name_scope(conv2.name):
        conv2.build(input_shape)
    conv2.W = conv1.W
    conv2._trainable_weights = []
    conv2._trainable_weights.append(conv2.W)

def L_Nested(beta_num, nested_dict, nExtraFeatures, networkSize, hidden_layers=1, train_betas=True, minima=None, logits_activation='softmax', dropout = 0.2, regularizer = None):
	""" L-MNL Model
		1) Main Input (X) follows the MNL architecture till Utilities
		2) Extra Input (Q) follows the DNN architecture till new feature
		3) Both terms are added to make Final Utilities
	"""
	choices_num = np.max(np.array([np.max(values) for values in nested_dict.values()])) + 1
	main_input = Input((beta_num, choices_num,1), name='Features')
	extra_input = Input((nExtraFeatures,1,1), name='Extra_Input')


	if minima is None:
		utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias=False, trainable=train_betas)(main_input)
	else:
		utilities = Conv2D(filters=1, kernel_size=[beta_num,1], strides=(1,1), padding='valid', name='Utilities',
			use_bias = False, weights=minima, trainable=train_betas)(main_input)
	#init = Constant(value=0.0001)
	dense = Conv2D(filters=networkSize, kernel_size=[nExtraFeatures, 1], activation='relu',
	#	padding='valid', name='Dense_NN_per_frame', kernel_regularizer=regularizer,  kernel_initializer=init)(extra_input)
		padding='valid', name='Dense_NN_per_frame', kernel_regularizer=regularizer)(extra_input)
	dropped = Dropout(dropout, name='Regularizer')(dense)

	x = dropped
	for i in range(hidden_layers-1):
	#	x = Dense(units=networkSize, activation='relu', name="Dense{}".format(i), kernel_initializer=init)(x)
		x = Dense(units=networkSize, activation='relu', name="Dense{}".format(i))(x)
		x = Dropout(dropout, name='Drop{}'.format(i))(x)
	dropped = x

	new_feature = Dense(units=choices_num, name="Output_new_feature")(dropped)
	#new_feature = Dense(units = choices_num, name = "Output_new_feature")(dense)

	new_featureR = Reshape([choices_num], name='Remove_Dim')(new_feature)
	utilitiesR = Reshape([choices_num], name='Flatten_Dim')(utilities)

	final_utilities = Add(name="New_Utility_functions")([utilitiesR, new_featureR])
	nests_gathered = []
	nests_softmax = []
	nests_multiplied = []
	utilities_divided = []

	exps = []
	sum = []
	logs = []
	for i, indices in enumerate(nested_dict.values()):
		nests_gathered.append(Lambda( lambda x:  K.tf.gather(x, indices, axis=1), name='gathered_Nests_{}'.format(i))(final_utilities))
		nest_constraint = MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0)

		# divide_layer = nestDivide(W_constraint=nest_constraint, name='nest_div_value_{}'.format(i))
		# utilities_divided.append(divide_layer(nests_gathered[i]))

		nest_constraint = MinMaxNorm(min_value=1.0, max_value=10.0, rate=1.0)
		multiply_layer = nestMultiply(W_constraint=nest_constraint, name='nest_value_{}'.format(i))
		nests_multiplied.append(multiply_layer(nests_gathered[i]))
		nests_softmax.append(Activation('softmax', name='nested_choices_{}'.format(i))(nests_multiplied[i]))


		#Inner probability:
		# nests_softmax.append(Activation('softmax', name='nested_choices_{}'.format(i))(nests_gathered[i]))
#		nests_softmax.append(Activation('softmax', name='nested_choices_{}'.format(i))(utilities_divided[i]))

		#Outer probability:
		nest_constraint = MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0)
#		exps.append(Activation('exponential', name='nests_exp{}'.format(i))(nests_gathered[i]))
		exps.append(Lambda( lambda x: K.exp(x), name='nests_exp{}'.format(i))(nests_gathered[i]))
		sum.append(Lambda(  lambda x: K.sum(x, axis=1, keepdims=True), name='nests_exp_sum_{}'.format(i))(exps[i]))
		logs.append(Lambda( lambda x: K.log(x), name='log_sum_{}'.format(i))(sum[i]))


		# multiply_layer = nestMultiply(W_constraint=nest_constraint, name='nest_value_{}'.format(i))
		# create_shared_weights(divide_layer, multiply_layer, logs[i]._keras_shape)
		# nests_multiplied.append(multiply_layer(logs[i]))

		nest_constraint = MinMaxNorm(min_value=1.0, max_value=10.0, rate=1.0)
		divide_layer = nestDivide(W_constraint=nest_constraint, name='nest_div_value_{}'.format(i))
		create_shared_weights(multiply_layer, divide_layer, logs[i]._keras_shape)
		utilities_divided.append(divide_layer(logs[i]))

#		nests_multiplied.append(nestMultiply(W_constraint=nest_constraint, name='nest_value_{}'.format(i))(logs[i]))
#		nests_multiplied.append(nestMultiply(name='nest_value_{}'.format(i))(logs[i]))
#		nests_multiplied.append(logs[i])

#	nest_utilities = Concatenate(axis=1)(nests_multiplied)
	nest_utilities = Concatenate(axis=1)(utilities_divided)
	nest_choice = Activation('softmax', name='main_choice')(nest_utilities)

	final_choices_logits = []
	nest_choice_value = []

	label_order = np.array([value for values in nested_dict.values() for value in values])

	for i, indices in enumerate(nested_dict.values()):
		nest_choice_value.append(Lambda( lambda x:  K.tf.gather(x, i, axis=1), name='single_nest_value_{}'.format(i))(nest_choice))
		final_choices_logits.append(Multiply(name='nests_probabilities_{}'.format(i))([nests_softmax[i], nest_choice_value[i]]))
	#	final_choices_logits.append(Multiply(name='nests_probabilities_{}'.format(i))([nests_softmax[i], nests_softmax[i]]))
		print(final_choices_logits[i].shape)
	all_logits = Concatenate(axis=1, name='final_probabilites')(final_choices_logits)
	probabilities = Lambda(lambda x: [x[:,label:label+1] for label in label_order])(all_logits)
	all_probabilites = Concatenate(axis=1, name='final_ordered_probabilites')(probabilities)

	#all_logits = Concatenate(axis=1)(nests_softmax)
	model = Model(inputs=[main_input,extra_input], outputs=all_probabilites)

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
