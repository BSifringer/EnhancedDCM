import numpy as np
import tensorflow as tf
import random
import shelve
import _pickle as pickle
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model, clone_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import np_utils, plot_model
from keras.losses import mean_squared_error
import keras.backend as K
from copy import copy, deepcopy


"""
	Various Tools for studying MNL or L-MNL models.
	- Gradients at input layer (saliency map)
	- Standard errors of Beta layer (from Hessian)
	- Sensitivity analysis
	and basic functions

	The main is used to investigate models one by one. 
"""


def get_inverse_Hessian(model, model_inputs, labels, layer_name='Utilities'):
	""" Note, in Tensorflow, gradient operator is normalized by length of data"""
	data_size = len(model_inputs[0])

# Get layer and gradient w.r.t. loss
	beta_layer = model.get_layer(layer_name)
	beta_gradient = K.gradients(model.total_loss, beta_layer.weights[0])[0]

# Get second order derivative operators (linewise of Hessian)
	Hessian_lines_op = {}
	for i in range(len(beta_layer.get_weights()[0])):
		Hessian_lines_op[i] = K.gradients(beta_gradient[i], beta_layer.weights[0])

# Define Functions that get operator values given inputed data
	input_tensors= model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
	get_Hess_funcs = {}
	for i in range(len(Hessian_lines_op)):
		get_Hess_funcs[i] = K.function(inputs=input_tensors, outputs=Hessian_lines_op[i])

# Line by line Hessian average multiplied by data length (due to automatic normalization)
	Hessian=[]
	func_inputs=[*[inputs for inputs in model_inputs], np.ones(data_size), labels, 0]
	for j in range(len(Hessian_lines_op)):
	    Hessian.append((np.array(get_Hess_funcs[j](func_inputs))))
	Hessian = np.squeeze(Hessian)*data_size

# The inverse Hessian:
	invHess = np.linalg.inv(Hessian)

	return invHess

def get_inputs_gradient(model, model_inputs, labels, inputs_indice=0):
	""" Get gradients on input layer with respect to prediction score (custom loss)"""

	def custom_loss(y_true, y_pred):
		return y_true*y_pred
		#return y_true*K.log(1-y_pred)
# Loss needs to be changed. Create a copy to avoid changing original model
	model_dummy = clone_model(model)
	model_dummy.set_weights(model.get_weights())
	model_dummy = Model(inputs=model_dummy.inputs, outputs=model_dummy.get_layer('New_Utility_functions').output)
	model_dummy.compile(loss=custom_loss, optimizer = 'adam', metrics = ['accuracy'])
	inputs_layer_placeholder = model_dummy.inputs[inputs_indice]

# Define the function
	input_tensors= model_dummy.inputs + model_dummy.sample_weights + model_dummy.targets + [K.learning_phase()]
	inputs_gradient = K.gradients(model_dummy.total_loss, inputs_layer_placeholder)
	get_gradient = K.function(inputs=input_tensors, outputs = inputs_gradient)

# Get inputs and call function
	func_inputs=[*[inputs for inputs in model_inputs], np.ones(len(model_inputs[0])), labels, 0]
	heatmap = np.squeeze(get_gradient(func_inputs))

	return heatmap


def get_likelihood_accuracy(model, model_inputs, labels):
	likelihood, accuracy = model.evaluate(model_inputs, labels, batch_size = 128, verbose = 0)
	return likelihood*(labels.shape[0]), accuracy*100


def get_betas(model, layer_name = 'Utilities'):
	beta_layer = model.get_layer(layer_name)
	return beta_layer.get_weights()[0].flatten()


def get_stds(model, model_inputs, labels, layer_name='Utilities'):
	""" Gets the diagonal of the inverse Hessian, square rooted """
	inv_Hess = get_inverse_Hessian(model, model_inputs, labels, layer_name)
	stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])]
	return np.array(stds).flatten()

def elasticity_sample_study(model, sample_input, inputs_indice, feature_indice, n=100, range=[-1,1]):
	points = np.linspace(range[0],range[1],n)
	predictions = []
	new_model = Model(inputs=model.inputs, outputs=model.get_layer('New_Utility_functions').output)
	new_model.compile(optimizer='adam', loss='categorical_crossentropy')
	for i in points:
		sample_input[inputs_indice][:,feature_indice:feature_indice+1] = i
		predictions.append(new_model.predict(sample_input))
	return np.squeeze(predictions)


def elasticity_study(model, model_inputs, inputs_indice, feature_indice, n=100, x_range=[-0.3,0.3]):
	points = np.linspace(x_range[0],x_range[1],n)
	elasticity = []
	new_model = Model(inputs=model.inputs, outputs=model.get_layer('New_Utility_functions').output)
	new_model.compile(optimizer='adam', loss='categorical_crossentropy')
	old_predictions = new_model.predict(model_inputs)
	choices = np.argmax(old_predictions, axis=1)
	old_count = [(choices==i).sum() for i in range(3)]

	for i in points:
		new_inputs=[]
		for j in range(len(model_inputs)):
			new_inputs.append(model_inputs[j][:].copy())
		new_inputs[inputs_indice][:,feature_indice:feature_indice+1] = (1+i) * model_inputs[inputs_indice][:,feature_indice:feature_indice+1]


		new_predicts = new_model.predict(new_inputs)
		new_choices = np.argmax(new_predicts, axis=1)
		change_count = [ (new_choices==i).sum() - old_count[i] for i in range(3)]
	#	percentages = (new_predicts-old_predictions)#/old_predictions
		elasticity.append(change_count)
	#	elasticity.append(percentages.mean(axis=0))

	return points, elasticity


def sensitivity_study(model, model_inputs, inputs_indice, feature_indice, n=100, x_range=[-0.3,0.3]):
	points = np.linspace(x_range[0],x_range[1],n)
	elasticity = []
	new_model = Model(inputs=model.inputs, outputs=model.get_layer('New_Utility_functions').output)
	new_model.compile(optimizer='adam', loss='categorical_crossentropy')
	old_predictions = new_model.predict(model_inputs)
	for i in points:
		new_inputs=[]
		for j in range(len(model_inputs)):
			new_inputs.append(model_inputs[j][:].copy())
		new_inputs[inputs_indice][:,feature_indice:feature_indice+1] = (1+i) * model_inputs[inputs_indice][:,feature_indice:feature_indice+1]

		change = new_model.predict(new_inputs) - old_predictions
		percentages = change/old_predictions
		elasticity.append(change.mean(axis=0))

	return points, elasticity


def class_maximization(model, n_iteration, n_class, train_start, extra_start, labels):
	inputs_indice=1
	def custom_loss(y_true, y_pred):
		return y_true*y_pred
		#return y_true*K.log(1-y_pred)
# Loss needs to be changed. Create a copy to avoid changing original model
	model_dummy = clone_model(model)
	model_dummy.set_weights(model.get_weights())
	model_dummy = Model(inputs=model_dummy.inputs, outputs=model_dummy.get_layer('New_Utility_functions').output)
	model_dummy.compile(loss=custom_loss, optimizer = 'adam', metrics = ['accuracy'])
	model_dummy = model
	#model_dummy.compile(loss='mse', optimizer = 'adam', metrics = ['accuracy'])

	inputs_layer_placeholder = model_dummy.inputs[inputs_indice]

# Define the function
	input_tensors= model_dummy.inputs + model_dummy.sample_weights + model_dummy.targets + [K.learning_phase()]
	inputs_gradient = K.gradients(model_dummy.total_loss, inputs_layer_placeholder)
#	inputs_gradient = K.gradients(model.outputs, inputs_layer_placeholder)
	get_gradient = K.function(inputs=input_tensors, outputs = inputs_gradient)


# Get inputs and call function
	select = np.zeros(labels[0].shape)
	select[n_class] = 1
	print(select)
	labels = np.array([select for label in labels])
	print(labels.shape)

	for i in range(n_iteration):
		func_inputs=[train_start, extra_start, np.ones(len(train_start)), np.expand_dims(select,0), 0]
		gradient = np.array(get_gradient(func_inputs)[0])
		extra_start = extra_start - gradient

	heatmap = np.squeeze(extra_start)

	print(model.evaluate([train_start,extra_start], np.expand_dims(select,0)))
	print(model_dummy.evaluate([train_start,extra_start],  np.expand_dims(select,0)))
	return heatmap



if __name__ == '__main__':
	print("--- Example with MNL over Toy function -----")

	model_name = 'generated_data/generated_MNLBetaFreeze.h5'
	train_name = 'generated_data/keras_input_generated_train.npy'
	
	model_name = '../generated_data/freeze/generated_EnhancedNNFreeze2.h5'
	train_name = '../generated_data/freeze/keras_input_generated_noASC_train.npy'
	extra_name = '../generated_data/freeze/keras_input_generated_noASC_train_extra.npy'
	model_name = '../generated_data/freeze/generated_EnhancedBetaFreezeextra.h5'
#	model_name = '../generated_data/freeze/generated_EnhancedNNFreeze.h5'

#	model_name = 'swissmetro_paper/Swissmetro_paper_Enhanced2layer100extra.h5'
#	train_name = 'swissmetro_paper/keras_input_Swissmetro_paper_train.npy'
#	extra_name = 'swissmetro_paper/keras_input_Swissmetro_paper_train_extra.npy'
#	
#	model_name = '../generated_data/poster/generated_data_Enhancedscan100extra.h5'
#	train_name = '../generated_data/poster/keras_input_generated_data_simple_train.npy'
#	extra_name = '../generated_data/poster/keras_input_generated_data_simple_train_extra.npy'
#
	model_name = '../generated_data/poster/generated_data_MNL_Full.h5'
	train_name = '../generated_data/poster/keras_input_generated_data_train.npy'
	extra_name = '../generated_data/poster/keras_input_generated_data_train_extra.npy'
	test_name = '../generated_data/poster/keras_input_generated_data_test.npy'
	extra_test_name = '../generated_data/poster/keras_input_generated_data_test_extra.npy'

#
#	model_name = '../generated_data/poster/generated_data_Enhancedscan100extra.h5'
#	train_name = '../generated_data/poster/keras_input_generated_data_2betas_simple_train.npy'
#	extra_name = '../generated_data/poster/keras_input_generated_data_2betas_simple_train_extra.npy'

	model = load_model(model_name)
	
	train_data = np.load(train_name)
	labels = train_data[:,-1,:]
	train_data = np.delete(train_data, -1, axis = 1)
	train_data = np.expand_dims(train_data, -1)
	

	extra_data = np.load(extra_name)
	extra_data = np.expand_dims(extra_data,-1)
	extra_data = np.expand_dims(extra_data,-1)

	model_inputs = [train_data]
	#model_inputs = [train_data, extra_data]
	layer_name = 'Utilities'
	inputs_indice = 0

	test_data = np.load(test_name)
	labels_test = test_data[:, -1, :]
	test_data = np.delete(test_data, -1, axis=1)
	test_data = np.expand_dims(test_data, -1)

	#invHess = get_inverse_Hessian(model, model_inputs, labels, layer_name)
	#print("The Hessian is Symmetric: \n {} \n".format(np.linalg.inv(invHess)))
	#print("The inverse has small diagonal terms: \n {} \n".format(invHess))


	#heatmap = get_inputs_gradient(model, model_inputs, labels, inputs_indice)

	#print("Sum of all abs. gradients per input/target :\n {} \n".format(np.sum(abs(heatmap), axis=0)))
	#print("Mean of gradients per input/target:\n {} \n".format(np.mean((heatmap), axis=0)))
	likelihood,_ = get_likelihood_accuracy(model, model_inputs, labels)
	likelihood_test, _ = get_likelihood_accuracy(model, [test_data], labels_test)
	print("Likelihood of model is {} for {} observations ".format(-likelihood, len(model_inputs[0])))
	print("Likelihood Test of model is {} for {} observations ".format(-likelihood_test, len(test_data)))
	print("Betas are {}".format(get_betas(model)))
	print("And STDS are {}".format(get_stds(model,model_inputs,labels)))
