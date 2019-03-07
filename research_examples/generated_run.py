from generated_data import data_manager as generatedDM
from keras import backend as K
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
	Run script for Synthetic Datset Experiments
	For each model: 
		- Define architecture (# X inputs, # Q inputs, model architecture)
		- Create input with keras_input()
		- Run the appropriate function below
		
	Due to Binary choice, we use binary crossentropy and sigmoid activation function for all models for extra simplicity
	(mathematically equivalent to 2 labels categorical crossentropy with softmax)

	Main() flags:
	------------
	neuron_scan:		Perform a achitectural scan of neurons on L-MNL
	monte_carlo: 		Perform Monte Carlo experiments with multiple Models (MNL and LMNL)
	Hruschka:			Perform Monte Carlo experiments for Hruschka Models
	noise_data:			Perform experiments with noisy input data (Monte Carlo)
	correlations:		Perform experiments with correlated data (Monte Carlo)
	unseen:				Perform experiments with omitted variables (Monte Carlo)

	run all with:
	python3 generated_run.py --scan --mc --mc_hr --corr --unseen
"""

parser = argparse.ArgumentParser(description='Choose Flags for training on experiments')

parser.add_argument('--scan', action='store_true', help='Runs MNL and multiple increasing size L-MNL on illustrative utility function')
parser.add_argument('--mc', action='store_true', help='Basic Monte Carlo Experiment on a set of models')
parser.add_argument('--mc_hr', action='store_true', help='Monte Carlo experiment for Hruschka models')
parser.add_argument('--corr', action='store_true', help='MC correlation experiment (big simulation)')
parser.add_argument('--unseen', action='store_true', help='MC experiment with unseen causalities for all models')
#parser.add_argument('--noise', action='store_true', help='MC with noise on MNL and L-MNL ') # research experiment

args = parser.parse_args()


neuron_scan = args.scan
monte_carlo = args.mc
Hruschka = args.mc_hr
correlations = args.corr
unseen = args.unseen
noise_data = False

choices_num = 1 
batchSize = 50


def GeneratedMNL(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart = '', saveName = '',
				 loss='binary_crossentropy', logits_activation='sigmoid'):

	nEpoch = 150
	betas, saveExtension = ru.runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, 
									filePart, saveName = saveName, loss=loss, logits_activation=logits_activation)
	K.clear_session()	#Avoid Keras Memory leak
	return betas, saveExtension

def GeneratedNN(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = False,
				filePart = '',  saveName = '',
				networkSize = 100, loss='binary_crossentropy', logits_activation='sigmoid'):

	nEpoch = 200
	saveExtension = ru.runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput,
							nExtraFeatures, filePart, saveName = saveName, networkSize = networkSize, loss=loss,
							logits_activation=logits_activation)
	K.clear_session()
	return saveExtension

def GeneratedMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput = False,
				   minima = None, train_betas = True, filePart = '', saveName = '', networkSize = 100,
				   loss='binary_crossentropy', logits_activation='sigmoid'):
	nEpoch = 200
	betas, saveExtension = ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name,  batchSize,
										extraInput, nExtraFeatures, minima, train_betas, filePart, saveName = saveName,
										networkSize = networkSize , loss=loss, logits_activation=logits_activation)
	K.clear_session()
	return betas, saveExtension


if __name__ == '__main__':

	extensions = ['_train', '_test']

	if neuron_scan:
		filePath = 'generated_data/'
		folderName = 'illustrate/'
		fileInputName = 'generated_0'
		list = [2, 5, 10, 15, 25, 50, 100, 200, 500, 1001, 2000]
		simpleArchitecture = True
		beta_num = 3
		nExtraFeatures = 3

		print("Simple Architecture MNL model")
		_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
		GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0] )
		generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set

		print("Neuron scan on L-MNL")
		lmnlArchitecture = True
		beta_num = 2
		_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], lmnlArchitecture = lmnlArchitecture) # create keras input for train set
		for i in list:
			print("----------- L-MNL with {} Neurons ------".format(i))
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = True, filePart = extensions[0], networkSize = i, saveName = "scan{}".format(i))
		generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1], lmnlArchitecture = lmnlArchitecture) # create keras input for test set

		print("Full MNL model")
		simpleArchitecture = False
		beta_num = 6
		nExtraFeatures = 0
		_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0]) # create keras input for train set
		GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0], saveName = '_Full')
		generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1]) # create keras input for train set

#---------------------- Monte Carlo Simulation for all models --------------------
	if monte_carlo:
		filePath = 'generated_data/'
		folderName = 'monte_carlo/'
		fileInputBase = 'generated'

		for i in range(100):
			print('----------- Monte Carlo Data Number {} --------'.format(i))
			fileInputName = fileInputBase +'_{}'.format(i)

			simpleArchitecture = False
			beta_num = 6 # ASC + 5 betas
			nExtraFeatures = 0
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
			
			print('-- Full MNL -- ')
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name,
											filePart = extensions[0], saveName = '_Full')
			#Full MNL and HyperParam with NN
			print('-- Full MNL Hybrid -- ')
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName,  beta_num, choices_num, nExtraFeatures,
											  train_data_name,  extraInput = False, filePart = extensions[0], networkSize = 100, saveName = '_Full')
			generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set

		# ------------------------------------------------------------------------
			simpleArchitecture = True
			beta_num = 3
			nExtraFeatures = 3
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set

			print('-- MNL Hybrid --')
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = False, filePart = extensions[0], networkSize = 100)
			print('-- MNL --')
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name,
											filePart = extensions[0])
			
			print('-- Our Model L-MNL --')
			lmnlArchitecture = True
			beta_num = 2
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0],
														   lmnlArchitecture = lmnlArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = True, filePart = extensions[0], networkSize = 100)
			generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set

			print('--True MNL---')
			trueArchitecture = True
			beta_num = 4
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], trueArchitecture = trueArchitecture)
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0], saveName = '_True')

	if Hruschka:
		# Models here are equivalent to Hruschka 04 and 07 when faced with binary crossentropy
		filePath = 'generated_data/'
		folderName = 'monte_carlo_Hruschka/'
		folderName_data = 'monte_carlo/'
		fileInputBase = 'generated'

		for i in range(100):
			print('----------- Hruschka MC Data Number {} --------'.format(i))

			fileInputName = fileInputBase +'_{}'.format(i)
			simpleArchitecture = False
			beta_num = 6 # ASC + 5 betas
			nExtraFeatures = 0
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName_data,fileInputName, filePart = extensions[0],
														   simpleArchitecture = simpleArchitecture) # create keras input for train set

			print('-- Hruschka Full MNL --')
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName,  beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = False, filePart = extensions[0], networkSize = 3,
											  saveName = '_Full')
			print('-- Hruschka2004 100 --')
			saveExtension = GeneratedNN(filePath+folderName, fileInputName,  beta_num, choices_num, nExtraFeatures,
										train_data_name, extraInput = False, filePart = extensions[0], networkSize = 100,
										saveName = '_Big_Full')
			print('-- Hruschka2004 3 --')
			saveExtension = GeneratedNN(filePath+folderName, fileInputName,  beta_num, choices_num, nExtraFeatures, train_data_name,
										   extraInput = False, filePart = extensions[0], networkSize = 3, saveName = '_Small_Full')
			generatedDM.keras_input(filePath+folderName_data,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set
		# ------------------------------------------------------------------------
			print('-- Hruschka MNL --')
			simpleArchitecture = True
			beta_num = 3
			nExtraFeatures = 3
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName_data,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = False, filePart = extensions[0], networkSize = 3)
			generatedDM.keras_input(filePath+folderName_data,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set


# ------ Scan testing correlation effects on MNL and LMNL
	if correlations:
		filePath = 'generated_data/'
		folderName = 'correlations/'
		fileInputBase = 'generated'

		for i in range(900):
			print('----------- Correlation Data Number {} --------'.format(i))
			fileInputName = fileInputBase +'_{}'.format(i)

			simpleArchitecture = False
			beta_num = 6 # ASC + 5 betas
			nExtraFeatures = 0
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
			generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set
			#Full MNL
			print('-- Full MNL -- ')
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0], saveName = '_Full')


			#MNL
			simpleArchitecture = True
			beta_num = 3
			nExtraFeatures = 3
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
			print('-- MNL --')
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0] )
			generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[1], simpleArchitecture = simpleArchitecture) # create keras input for test set

			#Our Model
			print('-- Our Model --')
			lmnlArchitecture = True
			beta_num = 2
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0],
														   lmnlArchitecture = lmnlArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = True, filePart = extensions[0], networkSize = 100)
			generatedDM.keras_input(filePath + folderName, fileInputName, filePart=extensions[1], lmnlArchitecture=lmnlArchitecture)  # create keras input for test set

# -------- Scan when model is missing Causalities ----- 
	if unseen:
		filePath = 'generated_data/'
		folderName = 'unseen/'
		fileInputBase = 'generated'
			
		for i in range(100):
			print('----------- Unseen Data Number {} --------'.format(i))
			fileInputName = fileInputBase +'_{}'.format(i)

			print('-- Full MNL -- ')
			simpleArchitecture = False
			beta_num = 6 # ASC + 5 betas
			nExtraFeatures = 0
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0],
														   simpleArchitecture = simpleArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name,
											filePart = extensions[0], saveName = '_Full')

			print('-- MNL --')
			simpleArchitecture = True
			beta_num = 3
			nExtraFeatures = 3
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0])

			#Our Model
			print('-- Our Model --')
			lmnlArchitecture = True
			beta_num = 2
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], lmnlArchitecture = lmnlArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  train_data_name, extraInput = True, filePart = extensions[0], networkSize = 100)


#------------ Experiment when model is faced with noisy data ------
	if noise_data:
		filePath = 'generated_data/'
		folderName = 'noise_input/'
		fileInputBase = 'generated'

		for i in range(100):
			print('----------- NOISE Data Number {} --------'.format(i))
			fileInputName = fileInputBase +'_{}'.format(i)

			print('-- MNL --')
			simpleArchitecture = True
			beta_num = 3
			nExtraFeatures = 3
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], simpleArchitecture = simpleArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart = extensions[0])


			print('-- Our Model --')
			lmnlArchitecture = True
			beta_num = 2
			_,_, train_data_name = generatedDM.keras_input(filePath+folderName,fileInputName, filePart = extensions[0], lmnlArchitecture = lmnlArchitecture) # create keras input for train set
			_, saveExtension = GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											  rain_data_name, extraInput = True, filePart = extensions[0], networkSize = 100)
