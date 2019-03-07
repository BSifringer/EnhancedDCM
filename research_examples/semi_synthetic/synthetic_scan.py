# Allows to run as main from any directory and import utility packages
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    print(path[0])
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-3])
    path.append(dir(parent))
    parent = '/'.join(splits[:-2])
    path.append(dir(parent))
    parent = '/'.join(splits[:-1])
    path.append(dir(parent))
    __package__ = "generated_data"

import pickle
from EnhancedDCM.utilities import grad_hess_utilities as ghu
from keras.models import load_model
from semi_synthetic import data_manager as dm
import numpy as np
from keras import backend as K

"""
    Creates + saves dictionary and prints results for semi-synthetic data models
    
    Main() Flags
    ------------
    illustrative:   models w. semi-synthetic data with low non-linear contribution
    power-log:      models w. semi-synthetic data with high non-linear contribution
"""


illustrative = False
power_log = True




def fetch_model(neuron, path, extend):
    """ Load models of swissmetro semi-synthetic """
    filename = "{}swissmetro{}{}.h5".format(path, neuron, extend)
    return load_model(filename)


def get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=False, lmnlArchitecture=False, write=False):
    """ Returns: inputs X, normalized inputs Q and labels"""
    inputs_labels, extra, _ = dm.keras_input(filePath, fileInputName, filePart, simpleArchitecture=simpleArchitecture,
                                             lmnlArchitecture=lmnlArchitecture, write=write)

    labels = np.load(filePath + 'synth' + filePart + '_labels.npy')
    inputs = np.expand_dims(inputs_labels, -1)

    extra = np.expand_dims(extra, -1)
    extra = np.expand_dims(extra, -1)
    extra = (extra-extra.mean(axis=0))/extra.std(axis=0)
    return inputs, extra, labels


def scan_iteration(cases, paths_extends, filePath, fileInputBase, std_flag=False):
    """
    Fills a dictionary with all important model statistics
    :param cases:           model name type
    :param paths_extends:   tuple with path to model and model saveName extension
    :param filePath:        labels filepath
    :param fileInputBase:   base name of labels and models
    :param std_flag:        when active, adds stds from Hessian when possible (slower process)
    :return: enyclopedia:   dictionary of model values
    """
    fileInputName = fileInputBase

    def get_model_inputs_labels(filePart):
        """ Get model inputs for each .dat for Train and Test """
        inputs, extra_input, labels = get_inputs_labels(filePath, fileInputName, filePart)
        inputs_simple, extra_input_simple, _ = get_inputs_labels(filePath, fileInputName, filePart,
                                                                 simpleArchitecture=True)
        inputs_lmnl, extra_input_lmnl, _ = get_inputs_labels(filePath, fileInputName, filePart, lmnlArchitecture=True)
        # Assign inputs to models
        # MNL models have single input in Input layer
        model_inputs = {case: [inputs] for case in cases if 'FULL' in case}
        model_inputs.update({case: [inputs_simple] for case in cases if 'FULL' not in case})
        model_inputs['HYBRID'] = [inputs_lmnl, extra_input_lmnl]
        return model_inputs, labels

    model_inputs, train_labels = get_model_inputs_labels('_train')
    model_test_inputs, test_labels = get_model_inputs_labels('_test')
    # Dict to save all values and plotting
    encyclopedia = {}

    for case in cases:
        NN_flag = 'HRUSCHKA' in case
        # Get model extension name and path to .dat file
        path, extend = paths_extends[case]
        model = fetch_model('', path, extend)
        if not NN_flag:
            betas = ghu.get_betas(model)
        likelihood_train, accuracy_train = ghu.get_likelihood_accuracy(model, model_inputs[case], train_labels)
        likelihood_test, accuracy_test = ghu.get_likelihood_accuracy(model, model_test_inputs[case], test_labels)
        # Getting STD is slow (1-3s), avoid if possible
        if std_flag and not NN_flag:
            stds = ghu.get_stds(model, model_inputs[case], train_labels)

        if not NN_flag:
            encyclopedia['betas_' + case] = betas
        encyclopedia['likelihood_train_' + case] = likelihood_train
        encyclopedia['likelihood_test_' + case] = likelihood_test
        encyclopedia['accuracy_train_' + case] = accuracy_train
        encyclopedia['accuracy_test_' + case] = accuracy_test
        if std_flag and not NN_flag:
            encyclopedia['stds_' + case] = stds

    K.clear_session()      # Fix Keras Memory Bug
    return encyclopedia


if __name__ == '__main__':
    """ Fills cases dictionnary, prints model information """
    if illustrative:
        folder = '../illustrative/'
    else:
        folder = '../power_log/'
    filePath = folder
    fileInputBase = 'swissmetro'

    cases = [
        'MNL',
        'FULL_MNL',
        'HYBRID',
    ]

    paths_extends = {
        'MNL': [folder, '_MNL'],
        'FULL_MNL': [folder, '_MNL_Full'],
        'HYBRID': [folder, '_Enhancedextra']
    }

    encyclopedia = {}
    encyclopedia = scan_iteration(cases, paths_extends, filePath, fileInputBase, std_flag=True)
    pickle.dump(encyclopedia, open('Encyclopedia_.p', 'wb'))
