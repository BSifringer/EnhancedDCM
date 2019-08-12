from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
import pickle
import os
import timeit

from guevara import data_manager as dm
import numpy as np
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-1])
    path.append(dir(parent))

import utilities.run_utils as ru
import argparse

"""
	Run script for guev Real Datset Experiments
	For each model:
		- Define architecture (# X inputs, # Q inputs, model architecture)
		- Create input with keras_input()
		- Run the appropriate function below

	Main() flags:
	------------
	models:		Estimates many models on guev dataset
	scan: 		Perform a architectural scan of neurons on L-MNL
"""


parser = argparse.ArgumentParser(description='Choose Flags for training on experiments')
parser.add_argument('--scan', action='store_true', help='Trains multiple L-MNL models of increasing size on guev')
parser.add_argument('--models', action='store_true', help='Trains a full set of models on guev')
parser.add_argument('--hyperParam', action='store_true', help='Hyper-Parameter Search')
parser.add_argument('--bests', action='store_true', help='Run best Models')
parser.add_argument('--process', type = int, help='Parallel-processing ID', default=1)
parser.add_argument('--max_process', type = int, help='Parallel-processing Total', default=1)

args = parser.parse_args()

models = args.models
scan = args.scan
hyperParam = args.hyperParam
bests = args.bests
process = args.process
max_process = args.max_process
choices_num = 2  # Train, SM, Car
batchSize = 200


def guevMNL(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart = '', saveName = '', betas_save = True,
				 loss='categorical_crossentropy', logits_activation='softmax', verbose=0, validationRatio = 0, callback = None, model_inputs = None):

	nEpoch = 50
	betas, saveExtension = ru.runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart, betas_save = betas_save,
		saveName = saveName, loss=loss, logits_activation=logits_activation, verbose=verbose, validationRatio=validationRatio, callback = callback, model_inputs = model_inputs)
	K.clear_session()	#Avoid Keras Memory leak
	return betas, saveExtension

def guevNN(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = False,
				filePart = '',  saveName = '',
				networkSize = 100, loss='categorical_crossentropy', logits_activation='softmax', verbose=2, validationRatio = 0, callback = None):

	nEpoch = 30
	saveExtension = ru.runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput,
							nExtraFeatures, filePart, saveName = saveName, networkSize = networkSize, loss=loss,
							logits_activation=logits_activation, verbose = verbose, validationRatio = validationRatio, callback = callback)
	K.clear_session()
	return saveExtension

def guevMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = False, hidden_layers = 1,
				   minima = None, train_betas = True, filePart = '', saveName = '', networkSize = 3,  betas_save = True, dropout=0.2,
				   loss='categorical_crossentropy', logits_activation='softmax', verbose = 0, validationRatio = 0, callback = None, model_inputs = None):
	nEpoch = 50
	betas, saveExtension = ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name,  batchSize,
										extraInput, nExtraFeatures, minima, train_betas, filePart, saveName = saveName,  betas_save = betas_save, model_inputs = model_inputs, hidden_layers = hidden_layers,
										networkSize = networkSize , loss=loss, logits_activation=logits_activation, verbose=verbose, validationRatio = validationRatio, callback = callback, dropout=dropout)
	K.clear_session()
	return betas, saveExtension


def guevHrusch(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
	nEpoch = 200

	saveExtension = ru.runHrusch(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
									 filePart, saveName=saveName)
	K.clear_session()
	return saveExtension

def guevHrusch07(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
	nEpoch = 200

	saveExtension = ru.runHrusch07(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
									 filePart, saveName=saveName)
	K.clear_session()
	return saveExtension


def normalize(data):
	return (data-data.mean(axis=0))/(data.std(axis=0))

class TestCallback(Callback):
    def __init__(self, test_data, two_inputs = True):
        self.test_data = test_data
        self.two_inputs = two_inputs

    def on_epoch_end(self, epoch, logs={}):
        x, y, labels = self.test_data
        if self.two_inputs:
            loss, acc = self.model.evaluate([x,y], labels, verbose=0)
        else:
            loss, acc = self.model.evaluate(y, labels, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

def transform_data(X_data, Q_data):
	test_labels = X_data[:,-1,:]
	X_data = np.delete(X_data, -1, axis = 1)

	X_data = np.expand_dims(X_data, -1)


	nExtraFeatures = Q_data[0].size
	Q_data = np.expand_dims(np.expand_dims(Q_data, -1),-1)

	#X_data = normalize(X_data)
	Q_data = normalize(Q_data)
	return [X_data, Q_data], test_labels


def ReturnTestData(test_data_name):
	test_data = np.load(test_data_name)
	extra_data = np.load(test_data_name[:-4] + '_extra.npy')
	return transform_data(test_data, extra_data)



if __name__ == '__main__':

	# splits data into train and test set
	# extensions = dm.train_test_split(filePath, seed = 32)

	filePath = 'guevara/'
	extensions = ['_train', '_test']
	#extensions = ['']


	if models:
		folderName = 'models/'
		fileInputName = 'guevara'

		# print("Full NN")
		# beta_num = 0
		# nExtraFeatures = 16
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], NNArchitecture=True)
		# guevNN(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, networkSize = 3, validationRatio = 0.11)

		# print("L-MNL 3beta")
		# beta_num = 3
		# nExtraFeatures = 13
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], lmnlArchitecture=True)
		# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, networkSize = 4, validationRatio = 0.11)

		print("MNL")
		beta_num = 4
		_, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], lmnlArchitecture=False)
		_, _, train_data_name,_,_  = dm.keras_input_scan(filePath+folderName, fileInputName, filePart=extensions[0], utility_indices = [ 1,0,0,0,0,0,1,0,0,0,1,0,1,0], write = True)
		guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart=extensions[0], validationRatio = 0.11, verbose =2)


		# print("L-MNL")
		# beta_num = 6
		# nExtraFeatures = 10
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0])
		# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, validationRatio = 0.11)

		# print("MNL")
		# beta_num = 6
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0])
		# guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart=extensions[0], validationRatio = 0.11)

	if scan:
		folderName = 'scan/'
		fileInputName = 'guevara'
		n_range = 100
		beta_num = 5
		choices_num = 2
		nExtraFeatures = 4

		for i in range(n_range):
			print( "-----------  ITERATION  {}  ---------------".format(i))
			train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True)
			test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[1], write = False)


			# # bestSave = ModelCheckpoint(filePath+folderName+fileInputName +'_MNL_endo_{}.h5'.format(i), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
			#
			# # # Run MNL and test
			# print("--- MNL ----")
			# guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, saveName='_endo_{}'.format(i), filePart=extensions[0],
			# 	   model_inputs = train_data, betas_save = True, verbose = 2)
			#
			# # # bestSave = ModelCheckpoint(filePath+folderName+fileInputName +'_LMNL_{}.h5'.format(i), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

			# print("--- L-MNL ----")
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='{}'.format(i), extraInput=True,
			# 			networkSize=5, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, dropout=0)

			guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='100_{}'.format(i), extraInput=True,
			 			networkSize=100, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2)

			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='2lay_{}'.format(i), extraInput=True,
			#  			networkSize=18, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, hidden_layers = 2)

			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='3lay_{}'.format(i), extraInput=True,
			#  			networkSize=13, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, hidden_layers = 3)

			# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True, latentArchitecture = True)
			# print("--- MNL Latent ----")
			# guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, saveName='_latent_{}'.format(i), filePart=extensions[0],
			# 	   betas_save = True, verbose = 2)
			#
			# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True, trueArchitecture = True)
			# print("--- MNL True ----")
			# guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, saveName='_true_{}'.format(i), filePart=extensions[0],
			# 	  betas_save = True, verbose = 2)
			#

			#train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True, layerArchitecture = True)
			#guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='3lay_special_{}'.format(i), extraInput=True,
			# 			networkSize=5, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, hidden_layers = 2)

			# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True,
			# 	weakArchitecture = True)

			# # Run MNL and test
			# print("--- MNL ----")
			# guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, saveName='_weak_{}'.format(i), filePart=extensions[0],
			# 	   model_inputs = train_data, betas_save = True, verbose = 2)

			# print("--- L-MNL ----")
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='_weak_{}'.format(i), extraInput=True,
			# 			networkSize=20, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2)

			# print("--- L-MNL 2lay ----")
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='2lay_weak_{}'.format(i), extraInput=True,
			# 			networkSize=20, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, hidden_layers = 2)


			# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True,
			# 	multipleArchitecture = True)

			# print("--- L-MNL ----")
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='_multi_{}'.format(i), extraInput=True,
			# 			networkSize=20, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2)

			# print("--- L-MNL 2lay ----")
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='2lay_multi_{}'.format(i), extraInput=True,
			# 			networkSize=10, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, hidden_layers = 2)

            #train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True,
            #	multipleWeakArchitecture = True)

            #print("---w L-MNL ----")
            #guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='_wmulti_{}'.format(i), extraInput=True,
			#			networkSize=20, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2)

            #print("---w L-MNL 2lay ----")
            #guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='2lay_wmulti_{}'.format(i), extraInput=True,
#			networkSize=20, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, hidden_layers = 2)

			# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, iteration = i, filePart=extensions[0] , write = True,
			# 	correlArchitecture = True)
			# #
			print("--- L-MNL ----")
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='_correl_{}'.format(i), extraInput=True,
			# 			networkSize=5, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2, dropout=0)
			# guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, saveName='_correl100_{}'.format(i), extraInput=True,
			#  			networkSize=100, model_inputs = [train_data, extra_data], betas_save = True, verbose = 2)


	if bests:
		folderName = 'bests/'
		fileInputName = 'guevara'
		utility = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, filePart=extensions[0], utility_indices = utility , write = True)
		test_data, extra_test_data, test_data_name, _, _  = dm.keras_input_scan(filePath+folderName, fileInputName, filePart=extensions[1], utility_indices = utility , write = False)
		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)
		bestSave = ModelCheckpoint(filePath+folderName+fileInputName +'_MNL_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

		# Run MNL and test
		guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart=extensions[0], validationRatio = 0.11, callback=[bestSave],
			   model_inputs = train_data, betas_save = False, verbose = 2)

		# The callback needs to Reboot:
		bestSave = ModelCheckpoint(filePath+folderName+fileInputName +'_LMNL_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, validationRatio = 0.11,
					networkSize=5, callback=[bestSave], betas_save = False, verbose = 2)

	if hyperParam:
		folderName = 'hyperParam/'
		fileInputName = 'guevara'
		features = 14
		neurons = [1,2,3,4,5,6,7]

		best = {
			'accuracy': [0, 0, 0],
			'loss':		[-100, -100, -100],
			'accuracy_models_neurons': ['','',''],
			'accuracy_models_utility': [[],[],[]],
			'accuracy_models_train': [0,0,0],
			'loss_models_neurons': ['','',''],
			'loss_models_utility': [[],[],[]],
			'loss_models_train': [0,0,0],
			'iteration': 0
		}

		temp_erase_command = "rm " + filePath+folderName+fileInputName +'_{}_{}'.format(process,max_process)+'_Temp.h5'
		def save_best(temp_model, train_model_inputs, test_model_inputs, train_labels, test_labels, neurons, utility):
			loss, accuracy = temp_model.evaluate(test_model_inputs, test_labels, verbose = 0)
			train_loss, minimum = temp_model.evaluate(train_model_inputs, train_labels, verbose = 0)
			def store_best(measure, name, train_measure):
				for i in range(3):
					if measure > best[name][i]:

						best[name].insert(i,measure)
						best[name].pop(-1)
						best[name+'_models_neurons'].insert(i, neurons)
						best[name+'_models_neurons'].pop(-1)
						best[name+'_models_utility'].insert(i, utility)
						best[name+'_models_utility'].pop(-1)
						best[name+'_models_train'].insert(i, train_measure)
						best[name+'_models_train'].pop(-1)
						temp_model.save(filePath+folderName+'{}_{}_'.format(process,max_process)+ fileInputName +'_{}_{}.h5'.format(name,i))
						break
			if minimum > 0.78:
				store_best(accuracy, 'accuracy', minimum)
				store_best(-loss, 'loss', -train_loss)
			os.system(temp_erase_command)

		start = timeit.default_timer()/3600
		n_iterations = pow(2,features)
		chunk = int(n_iterations/max_process)
		beginning = (process-1)*chunk + 1
		if process == max_process:
			end = n_iterations
		else:
			end = (process)*chunk + 1

		for i in range(beginning, end):
			elapsed = timeit.default_timer()/3600
			print('----- Utility  {}  out of  {} -----'.format(i, pow(2,features)))
			print(' Time Spend:  {0:.0f} h; Time Left:  {1:.0f} h;  Estimated Total: {2:.0f} h'.format(elapsed-start, (elapsed-start)*(chunk/(i-beginning+1)-1),  (elapsed-start)*(chunk/(i-beginning+1))) )
#		for i in tqdm(range(1,features)):
			# Create Unique selection of 14 variables thanks to bit strings
			grid_string = '{0:b}'.format(i)
			betas = [int(char) for char in grid_string]
			padded = [0 for i in range(features-len(grid_string))]
			padded.extend(betas)
			if len(padded) != features:
				print('ERROR?')

			train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input_scan(filePath+folderName, fileInputName, filePart=extensions[0], utility_indices = padded , write = False)
			test_data, extra_test_data, test_data_name, _, _  = dm.keras_input_scan(filePath+folderName, fileInputName, filePart=extensions[1], utility_indices = padded , write = False)
			train_model_inputs, train_labels = transform_data(train_data, extra_data)
			test_model_inputs, test_labels = transform_data(test_data, extra_test_data)
			bestSave = ModelCheckpoint(filePath+folderName+fileInputName +'_{}_{}'.format(process,max_process)+'_Temp.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

			# Run MNL and test
			guevMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart=extensions[0], validationRatio = 0.11, callback=[bestSave],
				   model_inputs = train_data, betas_save = False)
			temp_model = load_model(filePath+folderName+fileInputName+'_{}_{}'.format(process,max_process)+'_Temp.h5')
			save_best(temp_model, train_model_inputs[0], test_model_inputs[0], train_labels, test_labels, 0, padded)

			for neuron in neurons:
				# The callback needs to Reboot:
				bestSave = ModelCheckpoint(filePath+folderName+fileInputName +'_{}_{}'.format(process,max_process)+'_Temp.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

				guevMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, validationRatio = 0.11,
			 			networkSize=neuron, callback=[bestSave], model_inputs = [train_data, extra_data], betas_save = False)
				temp_model = load_model(filePath+folderName+fileInputName+'_{}_{}'.format(process,max_process)+'_Temp.h5')
				save_best(temp_model, train_model_inputs, test_model_inputs, train_labels, test_labels, neuron, padded)
			print(best['accuracy'])
			if i%10 == 0:
				best['iteration'] = i
				pickle.dump(best, open(filePath+folderName+'{}_{}_'.format(process,max_process)+'best_models.p', 'wb'))
		print(best)
		pickle.dump(best, open(filePath+folderName+'{}_{}_'.format(process,max_process)+'best_models.p', 'wb'))
