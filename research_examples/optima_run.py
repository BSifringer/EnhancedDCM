from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2
from keras.models import load_model
import pickle
import os
import timeit

from optima import data_manager as dm
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
	Run script for hil Real Datset Experiments
	For each model:
		- Define architecture (# X inputs, # Q inputs, model architecture)
		- Create input with keras_input()
		- Run the appropriate function below

	Main() flags:
	------------
	models:		Estimates many models on hil dataset
	scan: 		Perform a architectural scan of neurons on L-MNL
"""


parser = argparse.ArgumentParser(description='Choose Flags for training on experiments')
parser.add_argument('--scan', action='store_true', help='Trains multiple L-MNL models of increasing size on hil')
parser.add_argument('--models', action='store_true', help='Trains a full set of models on hil')
parser.add_argument('--spread', action='store_true', help='Trains chosen models 100 times to show optimum spread')
parser.add_argument('--experiment', action='store_true', help='Good L-MNL specification and MNL Dummy')
parser.add_argument('--hyperParam', action='store_true', help='Hyper-Parameter Search')
parser.add_argument('--bests', action='store_true', help='Run best Models')
parser.add_argument('--process', type = int, help='Parallel-processing ID', default=1)
parser.add_argument('--max_process', type = int, help='Parallel-processing Total', default=1)
parser.add_argument('--neurons', type = int, help='GPU_bug_re-run?', default=25)

args = parser.parse_args()

models = args.models
scan = args.scan
spread = args.spread
experiment = args.experiment
hyperParam = args.hyperParam
bests = args.bests
process = args.process
max_process = args.max_process
list = [args.neurons]
choices_num = 3  # PT, Car, Walk
batchSize = 200


def hilMNL(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart = '', saveName = '', betas_save = True,
				 loss='categorical_crossentropy', logits_activation='softmax', verbose=0, validationRatio = 0, callback = None, model_inputs = None):

	nEpoch = 80
	betas, saveExtension = ru.runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart, betas_save = betas_save,
		saveName = saveName, loss=loss, logits_activation=logits_activation, verbose=verbose, validationRatio=validationRatio, callback = callback, model_inputs = model_inputs)
	K.clear_session()	#Avoid Keras Memory leak
	return betas, saveExtension

def hilNN(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = False,
				filePart = '',  saveName = '', dropout=0.2, regularizer = None,
				networkSize = 100, loss='categorical_crossentropy', logits_activation='softmax', verbose=2, validationRatio = 0, callback = None):

	nEpoch = 80
	saveExtension = ru.runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput,
							nExtraFeatures, filePart, saveName = saveName, networkSize = networkSize, loss=loss, dropout=dropout, regularizer = regularizer,
							logits_activation=logits_activation, verbose = verbose, validationRatio = validationRatio, callback = callback)
	K.clear_session()
	return saveExtension

def hilMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = False, hidden_layers = 1,
				   minima = None, train_betas = True, filePart = '', saveName = '', networkSize = 3,  betas_save = True,
				   loss='categorical_crossentropy', logits_activation='softmax', verbose = 0, validationRatio = 0, callback = None, model_inputs = None, dropout=0.2, regularizer = None):
	nEpoch = 80
	betas, saveExtension = ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name,  batchSize,
										extraInput, nExtraFeatures, minima, train_betas, filePart, saveName = saveName,  betas_save = betas_save, model_inputs = model_inputs, hidden_layers = hidden_layers,
										networkSize = networkSize , loss=loss, logits_activation=logits_activation, verbose=verbose, validationRatio = validationRatio, callback = callback, dropout=dropout, regularizer=regularizer)
	K.clear_session()
	return betas, saveExtension


def hilHrusch(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
	nEpoch = 200

	saveExtension = ru.runHrusch(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
									 filePart, saveName=saveName)
	K.clear_session()
	return saveExtension

def hilHrusch07(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
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
	#Q_data = normalize(Q_data)
	return [X_data, Q_data], test_labels


def ReturnTestData(test_data_name):
	test_data = np.load(test_data_name)
	extra_data = np.load(test_data_name[:-4] + '_extra.npy')
	return transform_data(test_data, extra_data)



if __name__ == '__main__':

	# splits data into train and test set
	# extensions = dm.train_test_split(filePath, seed = 32)

	filePath = 'optima/'
	extensions = ['_train', '_test']
	#extensions = ['']



	if models:
		folderName = 'models3/'
		fileInputName = 'optima'

		# print("Full NN")
		# beta_num = 0
		# nExtraFeatures = 16
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], NNArchitecture=True)
		# hilNN(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, networkSize = 3, validationRatio = 0.11)

		# print("L-MNL 3beta")
		# beta_num = 3
		# nExtraFeatures = 13
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], lmnlArchitecture=True)
		# hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, networkSize = 4, validationRatio = 0.11)
		dm.train_test_split()
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], fullArchitecture = True, dummyArchitecture=True)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], fullArchitecture = True, dummyArchitecture=True)
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], fullArchitecture = False, dummyArchitecture=False)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], fullArchitecture = False, dummyArchitecture=False)

		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

		print("MNL")
		hilMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, model_inputs=train_data, filePart=extensions[0], saveName = '_dummy', verbose =2)
		mnl = load_model(filePath+folderName + fileInputName + '_MNL_dummy.h5')
		loss_mnl, accuracy_mnl = mnl.evaluate(test_model_inputs[0], test_labels)
		print(loss_mnl)
		print(accuracy_mnl)
		print(loss_mnl*1906*0.2)
		loss_mnl, accuracy_mnl = mnl.evaluate(train_model_inputs[0], train_labels)
		print(loss_mnl)
		print(accuracy_mnl)
		print(loss_mnl*1906*0.8)

		# print("L-MNL")
		# #hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
		# #	networkSize=20, hidden_layers=1, saveName='')
		#
		print("L-MNL Full")
		model_type = '_antonin_100'
		best_name = filePath+folderName+fileInputName +'_LMNL_best' + model_type+ '.h5'
		bestSave = ModelCheckpoint(best_name, monitor='val_acc', verbose=0,
					save_best_only=True, save_weights_only=False, mode='auto', period=1)
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
			filePart=extensions[0], fullArchitecture=False)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
				filePart=extensions[1], fullArchitecture=False)

		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

		hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
			networkSize=100, hidden_layers=1, saveName=model_type, validationRatio=0.0, callback = [bestSave], dropout=0.3, regularizer=l2(0.2) )
		# best_model = load_model(best_name)
		#
		# loss, accuracy = best_model.evaluate(train_model_inputs,train_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.8)
		# loss, accuracy = best_model.evaluate(test_model_inputs,test_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.2)


		lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ 'extra.h5')
		loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.8)
		loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.2)

		print("L-MNL Full")
		model_type = '_antonin_100_simple'
		best_name = filePath+folderName+fileInputName +'_LMNL_best' + model_type+ '.h5'
		bestSave = ModelCheckpoint(best_name, monitor='val_acc', verbose=0,
					save_best_only=True, save_weights_only=False, mode='auto', period=1)
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
			filePart=extensions[0], simpleArchitecture=True)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
				filePart=extensions[1], simpleArchitecture=True)

		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

		hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
			networkSize=100, hidden_layers=1, saveName=model_type, validationRatio=0.0, callback = [bestSave], dropout=0.3, regularizer=l2(0.2) )
		# best_model = load_model(best_name)
		#
		# loss, accuracy = best_model.evaluate(train_model_inputs,train_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.8)
		# loss, accuracy = best_model.evaluate(test_model_inputs,test_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.2)
		#

		lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ 'extra.h5')
		loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.8)
		loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.2)


		# print("L-MNL Full")
		# model_type = '_full_15'
		# best_name = filePath+folderName+fileInputName +'_LMNL_best' + model_type+ '.h5'
		# bestSave = ModelCheckpoint(best_name, monitor='val_acc', verbose=0,
		# 			save_best_only=True, save_weights_only=False, mode='auto', period=1)
		# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
		# 	filePart=extensions[0], fullArchitecture=True)
		# test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
		# 		filePart=extensions[1], fullArchitecture=True)
		#
		# train_model_inputs, train_labels = transform_data(train_data, extra_data)
		# test_model_inputs, test_labels = transform_data(test_data, extra_test_data)
		#
		# hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
		# 	networkSize=15, hidden_layers=1, saveName=model_type, validationRatio=0.2, callback = [bestSave], dropout=0.5, regularizer=l2(0.5) )
		# best_model = load_model(best_name)
		#
		# loss, accuracy = best_model.evaluate(train_model_inputs,train_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.8)
		# loss, accuracy = best_model.evaluate(test_model_inputs,test_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.2)
		#
		#
		# lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ 'extra.h5')
		# loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.8)
		# loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906*0.2)
		#
		# print("L-MNL Full full dataset")
		# model_type = '_full_15'
		# best_name = filePath+folderName+fileInputName +'_LMNL_best' + model_type+ '.h5'
		# bestSave = ModelCheckpoint(best_name, monitor='val_acc', verbose=0,
		# 			save_best_only=True, save_weights_only=False, mode='auto', period=1)
		# train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
		# 	filePart='', fullArchitecture=True)
		#
		# train_model_inputs, train_labels = transform_data(train_data, extra_data)
		#
		# hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
		# 	networkSize=15, hidden_layers=1, saveName=model_type, validationRatio=0.2, callback = [bestSave], dropout=0.5, regularizer=l2(0.5) )
		# best_model = load_model(best_name)
		#
		# loss, accuracy = best_model.evaluate(train_model_inputs,train_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906)
		#
		# lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ 'extra.h5')
		# loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
		# print(loss)
		# print(accuracy)
		# print(loss * 1906)


		# print("MNL")
		# beta_num = 6
		# _, _, train_data_name = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0])
		# hilMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart=extensions[0], validationRatio = 0.11)


		print("NN Full")
		model_type = '_antonin_100_NN'
		best_name = filePath+folderName+fileInputName +'_LMNL_best' + model_type+ '.h5'
		bestSave = ModelCheckpoint(best_name, monitor='val_acc', verbose=0,
					save_best_only=True, save_weights_only=False, mode='auto', period=1)
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
			filePart=extensions[0], simpleArchitecture=True)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
				filePart=extensions[1], simpleArchitecture=True)

		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

		hilNN(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
			networkSize=100, saveName=model_type, validationRatio=0.2, callback = [bestSave], dropout=0.3, regularizer=l2(0.2) )
		best_model = load_model(best_name)

		loss, accuracy = best_model.evaluate(train_model_inputs[1],train_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.8)
		loss, accuracy = best_model.evaluate(test_model_inputs[1],test_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.2)


		lmnl = load_model(filePath+folderName + fileInputName +'_NN' +model_type+ 'extra.h5')
		loss, accuracy = lmnl.evaluate(train_model_inputs[1],train_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.8)
		loss, accuracy = lmnl.evaluate(test_model_inputs[1],test_labels)
		print(loss)
		print(accuracy)
		print(loss * 1906*0.2)



	if scan:
		folderName = 'scan2/'
		fileInputName = 'optima'
		n_range = 10
		list = [1, 5, 10, 15, 25, 50, 100, 200, 500, 1001, 2000]
		dm.train_test_split()
#		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0])
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], fullArchitecture=True)

		for i in list:
			for j in range(n_range):
				print("L-MNL with {} neurons, iteration {}".format(i,j))
				hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
				networkSize=i, hidden_layers=1, saveName="_1lay_{}_{}".format(i,j),dropout=0.5, regularizer=l2(0.5))

				print("L-MNL 2 with {} neurons".format(i))
				hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
				networkSize=i, hidden_layers=2, saveName="_2lay_{}_{}".format(i,j),dropout=0.5, regularizer=l2(0.5) )

	if spread:
		folderName = 'spread2/'
		fileInputName = 'optima'
		n_range = 100
		# n_range = 1
		dict = {
			'MNL':{},
			'MNL_dummy':{},
			'L-MNL100':{},
			'L-MNL100_simple':{},
			'L-MNL30':{},
			'L-MNL30_simple':{},
			'L-MNL5':{},
			'L-MNL5_simple':{},
			'NN100':{},
			'NN30':{},
			'NN5':{},
			'L-MNL_in':{},
			'MNL_in_dummy':{}
		}

		for key in dict.keys():
			dict[key] = {
			'loss_train':[],
			'loss_test':[],
			'acc_train':[],
			'acc_test':[]
			}

		dm.train_test_split(seed=10)

		for j in range(n_range):
			train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], fullArchitecture = False, dummyArchitecture=False)
			test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], fullArchitecture = False, dummyArchitecture=False)

			train_model_inputs, train_labels = transform_data(train_data, extra_data)
			test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

			print('-------------- This is {}th iteration -----------'.format(j))

			print("MNL")
			case = 'MNL'
			hilMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, model_inputs=train_data, filePart=extensions[0], saveName = '_{}'.format(j), verbose =2)
			mnl = load_model(filePath+folderName + fileInputName + '_MNL_{}.h5'.format(j))
			loss_mnl, accuracy_mnl = mnl.evaluate(test_model_inputs[0], test_labels)
			print(loss_mnl)
			print(accuracy_mnl)
			dict[case]['loss_test'].append(loss_mnl)
			dict[case]['acc_test'].append(accuracy_mnl)

			loss_mnl, accuracy_mnl = mnl.evaluate(train_model_inputs[0], train_labels)
			print(loss_mnl)
			print(accuracy_mnl)
			dict[case]['loss_train'].append(loss_mnl)
			dict[case]['acc_train'].append(accuracy_mnl)

			print("L-MNL_experiment")
			case = 'L-MNL_in'
			model_type = '_experiment'
			train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], inArchitecture=True)
			test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], inArchitecture=True)

			train_model_inputs, train_labels = transform_data(train_data, extra_data)
			test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

			hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
					networkSize=10, hidden_layers=1, saveName=model_type+'_{}_'.format(j), validationRatio=0.0, dropout=0.2, regularizer=l2(0.2) )

			lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ '_{}_extra.h5'.format(j))
			loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
			print(loss)
			print(accuracy)
			dict[case]['loss_train'].append(loss)
			dict[case]['acc_train'].append(accuracy)

			loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
			print(loss)
			print(accuracy)
			dict[case]['loss_test'].append(loss)
			dict[case]['acc_test'].append(accuracy)


			print("MNL_dummy")
			case = 'MNL_dummy'
			train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], dummyArchitecture=True)
			test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], dummyArchitecture=True)

			train_model_inputs, train_labels = transform_data(train_data, extra_data)
			test_model_inputs, test_labels = transform_data(test_data, extra_test_data)
			hilMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, model_inputs=train_data, filePart=extensions[0], saveName = '_dummy_{}'.format(j), verbose =2)
			mnl = load_model(filePath+folderName + fileInputName + '_MNL_dummy_{}.h5'.format(j))
			loss_mnl, accuracy_mnl = mnl.evaluate(test_model_inputs[0], test_labels)
			print(loss_mnl)
			print(accuracy_mnl)
			dict[case]['loss_test'].append(loss_mnl)
			dict[case]['acc_test'].append(accuracy_mnl)

			loss_mnl, accuracy_mnl = mnl.evaluate(train_model_inputs[0], train_labels)
			print(loss_mnl)
			print(accuracy_mnl)
			dict[case]['loss_train'].append(loss_mnl)
			dict[case]['acc_train'].append(accuracy_mnl)

			print("MNL_in_dummy")
			case = 'MNL_in_dummy'
			train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], inArchitecture=True, dummyArchitecture=True)
			test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], inArchitecture=True, dummyArchitecture=True)

			train_model_inputs, train_labels = transform_data(train_data, extra_data)
			test_model_inputs, test_labels = transform_data(test_data, extra_test_data)
			hilMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, model_inputs=train_data, filePart=extensions[0], saveName = '_in_dummy_{}'.format(j), verbose =2)
			mnl = load_model(filePath+folderName + fileInputName + '_MNL_in_dummy_{}.h5'.format(j))
			loss_mnl, accuracy_mnl = mnl.evaluate(test_model_inputs[0], test_labels)
			print(loss_mnl)
			print(accuracy_mnl)
			dict[case]['loss_test'].append(loss_mnl)
			dict[case]['acc_test'].append(accuracy_mnl)

			loss_mnl, accuracy_mnl = mnl.evaluate(train_model_inputs[0], train_labels)
			print(loss_mnl)
			print(accuracy_mnl)
			dict[case]['loss_train'].append(loss_mnl)
			dict[case]['acc_train'].append(accuracy_mnl)


			for neurons in [5, 30,100]:

				print('-------------- This is {}th iteration -----------'.format(j))

				if neurons == 30:
					dropout = 0.3
					regularizer = 0.5
				elif neurons == 100:
					dropout = 0.3
					regularizer = 0.5
				else:
					dropout = 0.1
					regularizer = 0.2


				print('L-MNL{}'.format(neurons))
				case = 'L-MNL{}'.format(neurons)
				model_type = '_antonin_{}'.format(neurons)

				train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
					filePart=extensions[0], fullArchitecture=False)
				test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
						filePart=extensions[1], fullArchitecture=False)

				train_model_inputs, train_labels = transform_data(train_data, extra_data)
				test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

				hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
					networkSize=neurons, hidden_layers=1, saveName=model_type+'_{}_'.format(j), validationRatio=0.0, dropout=dropout, regularizer=l2(regularizer) )

				lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ '_{}_extra.h5'.format(j))
				loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
				print(loss)
				print(accuracy)
				dict[case]['loss_train'].append(loss)
				dict[case]['acc_train'].append(accuracy)

				loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
				print(loss)
				print(accuracy)
				dict[case]['loss_test'].append(loss)
				dict[case]['acc_test'].append(accuracy)

				print("L-MNL Simple")
				case  = 'L-MNL{}_simple'.format(neurons)
				model_type = '_antonin_simple_{}'.format(neurons)

				train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
					filePart=extensions[0], simpleArchitecture=True)
				test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
						filePart=extensions[1], simpleArchitecture=True)

				train_model_inputs, train_labels = transform_data(train_data, extra_data)
				test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

				hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
					networkSize=neurons, hidden_layers=1, saveName=model_type+'_{}_'.format(j), validationRatio=0.0, dropout=dropout, regularizer=l2(regularizer) )

				lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ '_{}_extra.h5'.format(j))
				loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
				print(loss)
				print(accuracy)
				dict[case]['loss_train'].append(loss)
				dict[case]['acc_train'].append(accuracy)

				loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
				print(loss)
				print(accuracy)
				dict[case]['loss_test'].append(loss)
				dict[case]['acc_test'].append(accuracy)


				print("NN Full")
				case = 'NN{}'.format(neurons)
				model_type = '_antonin_NN{}'.format(neurons)

				train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
					filePart=extensions[0], NNArchitecture=True)
				test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName,
						filePart=extensions[1], NNArchitecture=True)

				train_model_inputs, train_labels = transform_data(train_data, extra_data)
				test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

				hilNN(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
					networkSize=neurons, saveName=model_type+'_{}_'.format(j), dropout=dropout, regularizer=l2(regularizer) )

				lmnl = load_model(filePath+folderName + fileInputName +'_NN' +model_type+ '_{}_extra.h5'.format(j))
				loss, accuracy = lmnl.evaluate(train_model_inputs[1],train_labels)
				print(loss)
				print(accuracy)
				dict[case]['loss_train'].append(loss)
				dict[case]['acc_train'].append(accuracy)
				loss, accuracy = lmnl.evaluate(test_model_inputs[1],test_labels)
				print(loss)
				print(accuracy)
				dict[case]['loss_test'].append(loss)
				dict[case]['acc_test'].append(accuracy)
			if j%10 ==0:
				pickle.dump(dict, open(filePath+folderName+'spread_temp.p', 'wb'))

		pickle.dump(dict, open(filePath+folderName+'spread.p', 'wb'))

	if experiment:
		dm.train_test_split(seed = 6)
		fileInputName = 'optima'
		folderName = 'experiment/'
		print("L-MNL_experiment")
		case = 'L-MNL_in'
		model_type = '_experiment'
		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], inArchitecture=True)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], inArchitecture=True)

		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

		hilMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose = 2,
				networkSize=5, hidden_layers=1, saveName=model_type, validationRatio=0.2, dropout=0.2, regularizer=l2(0.5) )

		lmnl = load_model(filePath+folderName + fileInputName +'_Enhanced' +model_type+ 'extra.h5')
		loss, accuracy = lmnl.evaluate(train_model_inputs,train_labels)
		print(loss)
		print(accuracy)
		# dict[case]['loss_train'].append(loss)
		# dict[case]['acc_train'].append(accuracy)

		loss, accuracy = lmnl.evaluate(test_model_inputs,test_labels)
		print(loss)
		print(accuracy)
		# dict[case]['loss_test'].append(loss)
		# dict[case]['acc_test'].append(accuracy)

		train_data, extra_data, train_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[0], inArchitecture=True, dummyArchitecture=True)
		test_data, extra_test_data, test_data_name, beta_num, nExtraFeatures = dm.keras_input(filePath+folderName, fileInputName, filePart=extensions[1], inArchitecture=True, dummyArchitecture=True)

		train_model_inputs, train_labels = transform_data(train_data, extra_data)
		test_model_inputs, test_labels = transform_data(test_data, extra_test_data)

		hilMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, model_inputs=train_data, filePart=extensions[0], saveName = '_dummy', verbose =2)
		mnl = load_model(filePath+folderName + fileInputName + '_MNL_dummy.h5')
		loss_mnl, accuracy_mnl = mnl.evaluate(test_model_inputs[0], test_labels)
		print(loss_mnl)
		print(accuracy_mnl)
		print(loss_mnl*1906*0.2)
		loss_mnl, accuracy_mnl = mnl.evaluate(train_model_inputs[0], train_labels)
		print(loss_mnl)
		print(accuracy_mnl)
		print(loss_mnl*1906*0.8)
