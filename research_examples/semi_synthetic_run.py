import numpy as np
import argparse
from semi_synthetic import data_manager as semiDM
from keras.optimizers import RMSprop, Adam, SGD
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-1])
    path.append(dir(parent))

import utilities.models as mdl
import utilities.train_utils as tu
from keras import backend as K

## Here we can create any runs after data has been managed:
# Choose parameters
# Get a model
# Choose an optimizer and compile the model.
# Train with same or extra input ( based on model and problem set )


parser = argparse.ArgumentParser(description='Choose Flags for training on experiments. Default: Highly non-Linear Utility Function with Real Data')
parser.add_argument('--illustrative', action='store_true', help='Fit a simple Synthetic Utility Function with Real Data')
args = parser.parse_args()


illustrative = args.illustrative
power_log = not illustrative

batchSize = 50

def train_sameInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '',
                    saveExtension = '', filePart = '', callback = None, validationRatio = 0, verbose = 0 ):

    train_data = np.load(train_data_name)
    train_labels = np.load(filePath + 'synth' + filePart + '_labels.npy')

    train_data = np.expand_dims(train_data, -1)

    tu.fitModel(train_data, train_labels, nEpoch,batchSize,compiledModel,callback,validationRatio)

    compiledModel.save(filePath + fileInputName +'_' + saveExtension + '.h5')



def train_extraInput(fileInputName, nEpoch, compiledModel, train_data_name, batchSize = 50, filePath = '',
                     saveExtension = '', filePart = '', callback = None, validationRatio = 0, NN = False , verbose = 0):
    train_data = np.load(train_data_name)
    train_labels = np.load(filePath + 'synth' + filePart + '_labels.npy')
    train_data = np.expand_dims(train_data, -1)


    extra_data = np.load(train_data_name[:-4] + '_extra.npy')
    extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

    extra_data = (extra_data - extra_data.mean(axis=0))/extra_data.std(axis=0)

    if not NN:
        tu.fitModel([train_data, extra_data], train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, verbose = verbose)
    else:
        tu.fitModel(extra_data, train_labels, nEpoch, batchSize,compiledModel,callback,validationRatio, verbose = verbose)

    compiledModel.save(filePath + fileInputName + '_' + saveExtension + '.h5')

def GeneratedMNL(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName='',
                 loss='categorical_crossentropy', logits_activation='softmax'):
    nEpoch = 150

    saveExtension = 'MNL' + saveName

    model = mdl.MNL(beta_num, choices_num, logits_activation=logits_activation)
    optimizer = Adam(clipnorm=50.)
    model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)

    train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath=filePath,
                          saveExtension=saveExtension, filePart=filePart)

    betas = tu.saveBetas(fileInputName, model, filePath=filePath, saveExtension=saveExtension)

    K.clear_session()
    return betas, saveExtension
#    return ru.runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, filePart,
#                     saveName=saveName, loss=loss, logits_activation=logits_activation)


def GeneratedNN(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=False,
                filePart='', saveName='', loss='categorical_crossentropy', logits_activation='softmax'):
    nEpoch = 400

    saveExtension = 'NN' + saveName
    if extraInput:
        saveExtension = saveExtension + 'extra'
        model = mdl.denseNN_extra(beta_num, choices_num, nExtraFeatures, networkSize=networkSize,
                                  logits_activation=logits_activation)
    else:
        model = mdl.denseNN(beta_num, choices_num, networkSize=networkSize, logits_activation=logits_activation)

    optimizer = Adam(clipnorm=50.)
    model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)
    if extraInput:
        train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath=filePath,
                            saveExtension=saveExtension, filePart=filePart, NN=True)
    else:
        train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath=filePath,
                           saveExtension=saveExtension, filePart=filePart)

    # plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')
    K.clear_session()
    return saveExtension

#    return ru.runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput,
#                    nExtraFeatures, filePart, saveName=saveName, loss=loss, logits_activation=logits_activation)


def GeneratedMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=False,
                   minima=None, train_betas=True, filePart='', saveName='', networkSize=16, verbose=0,
                   loss='categorical_crossentropy', logits_activation='softmax'):
    nEpoch = 200

    saveExtension = 'Enhanced' + saveName
    if extraInput:
        saveExtension = saveExtension + 'extra'
        model = mdl.enhancedMNL_extraInput(beta_num, choices_num, nExtraFeatures, networkSize=networkSize,
                                           minima=minima, train_betas=train_betas,
                                           logits_activation=logits_activation)
    else:
        model = mdl.enhancedMNL_sameInput(beta_num, choices_num, minima=minima, train_betas=train_betas,
                                          logits_activation=logits_activation)

    # optimizer = SGD(momentum = 0.2, decay = 0.001)
    optimizer = Adam(clipnorm=100.)
    model.compile(optimizer=optimizer, metrics=["accuracy"], loss=loss)

    if extraInput:
        train_extraInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath=filePath,
                         saveExtension=saveExtension, filePart=filePart, verbose=verbose)
    else:
        train_sameInput(fileInputName, nEpoch, model, train_data_name, batchSize, filePath=filePath,
                        saveExtension=saveExtension, filePart=filePart, verbose=verbose)

    betas = tu.saveBetas(fileInputName, model, filePath=filePath, saveExtension=saveExtension)

    # plot_model(model, to_file = filePath + fileInputName+'_'+saveExtension+'.png')

    K.clear_session()
    return betas, saveExtension
#    return ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize, extraInput,
#                       nExtraFeatures, minima, train_betas, filePart, saveName=saveName, networkSize=networkSize,
#                       verbose=verbose, loss=loss, logits_activation=logits_activation)


if __name__ == '__main__':
    choices_num = 3
    filePath = 'semi_synthetic/'
    extensions = ['_train', '_test']
    fileInputName = 'swissmetro'

    if illustrative:
        folderName = 'illustrative/'
    else:
        folderName = 'power_log/'

    print('MNL')
    simpleArchitecture=True
    beta_num = 4
    _, _, train_data_name = semiDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
                                               simpleArchitecture=simpleArchitecture)
    GeneratedMNL(filePath+folderName, fileInputName, beta_num, choices_num, train_data_name,
                 filePart=extensions[0])
    print('MNL_Full ')
    simpleArchitecture = False
    beta_num = 9
    _, _, train_data_name = semiDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
                                               simpleArchitecture=simpleArchitecture)
    GeneratedMNL(filePath+folderName, fileInputName, beta_num, choices_num, train_data_name,
                 filePart=extensions[0], saveName="_Full")

    print("L-MNL")
    lmnlArchitecture = True
    extraInput = True
    beta_num = 2
    nExtraFeatures =  14 #17  # fix data manager scale + normalize + features etc.
    _, _, train_data_name = semiDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
                                               lmnlArchitecture=lmnlArchitecture)  # create keras input for train set
    GeneratedMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
                   train_data_name, extraInput=extraInput,
                   filePart=extensions[0], networkSize=100)
