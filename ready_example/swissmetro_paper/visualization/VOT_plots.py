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
from EnhancedDCM.utilities import models as mdl
from keras.models import load_model
from swissmetro_paper import data_manager as dm
import numpy as np
from keras import backend as K
from keras.optimizers import Adam



"""
    Saves in file and Writes results of all Models specified in "Cases"
"""

def fetch_model(neuron, path, extend, case):
    filename = "{}swissmetro{}{}.h5".format(path, neuron, extend)
    if 'NEST' not in case:
        return load_model(filename)
    else:
        if case == 'FULL_NEST':
            beta_num=9
            nExtraFeatures = 1
        if case == 'LMNL_FULL_NEST':
            beta_num=9
            nExtraFeatures = 8
        if case == 'LMNL_NEST':
            beta_num=3
            nExtraFeatures = 12
        nested_dict = {0:[0,2], 1: [1]}
        model = mdl.L_Nested(beta_num, nested_dict, nExtraFeatures, networkSize = 100)
        optimizer = Adam()
        model.compile(optimizer=optimizer, metrics=["accuracy"], loss='categorical_crossentropy')
        model.load_weights(filename)
        return model

def get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=False, lmnlArchitecture=False, correlArchitecture=False,
 nnArchitecture=False, subArchitecture=False, nlArchitecture=False, dummyArchitecture=False, write=False):
    inputs_labels, extra, _ = dm.keras_input(filePath, fileInputName, filePart, simpleArchitecture=simpleArchitecture, nnArchitecture=nnArchitecture, subArchitecture=subArchitecture,
                                             lmnlArchitecture=lmnlArchitecture, correlArchitecture=correlArchitecture, nlArchitecture=nlArchitecture, dummyArchitecture=dummyArchitecture, write=write)

    labels = inputs_labels[:, -1, :]
    inputs = np.delete(inputs_labels, -1, axis=1)
    inputs = np.expand_dims(inputs, -1)

    if subArchitecture:
        extra = [np.expand_dims(np.expand_dims(subset, -1),-1) for subset in extra]
    else:
        extra = np.expand_dims(extra, -1)
        extra = np.expand_dims(extra, -1)
    #extra = (extra-extra.mean(axis=0))/extra.std(axis=0)
    return inputs, extra, labels


def scan_iteration(cases, paths_extends, filePath, fileInputBase, encyclopedia, std_flag=False):
    fileInputName = fileInputBase

    def get_model_inputs_labels(filePart):
        # Get model inputs for each .dat for Train and Test
        inputs, extra_input, labels = get_inputs_labels(filePath, fileInputName, filePart)
        inputs_simple, extra_input_simple, _ = get_inputs_labels(filePath, fileInputName, filePart,
                                                                 simpleArchitecture=True)
        inputs_lmnl, extra_input_lmnl, _ = get_inputs_labels(filePath, fileInputName, filePart, lmnlArchitecture=True)
        # inputs_lmnl_cor, extra_input_lmnl_cor, _ = get_inputs_labels(filePath, fileInputName, filePart, correlArchitecture=True)
        inputs_lmnl_cor, extra_input_lmnl_cor, _ = get_inputs_labels(filePath, fileInputName, filePart, subArchitecture=True, lmnlArchitecture=True)
        inputs_lmnl_sub, extra_input_lmnl_sub, _ = get_inputs_labels(filePath, fileInputName, filePart, subArchitecture=True, correlArchitecture=True)
        inputs_lmnl_subc, extra_input_lmnl_subc, _ = get_inputs_labels(filePath, fileInputName, filePart, subArchitecture=True,  lmnlArchitecture=True)
        inputs_lmnl_subnl, extra_input_lmnl_subnl, _ = get_inputs_labels(filePath, fileInputName, filePart, nlArchitecture=True)
        inputs_NN, extra_input_NN, _ = get_inputs_labels(filePath, fileInputName, filePart, nnArchitecture=True)
        inputs_Dumy, extra_input_Dummy, _ = get_inputs_labels(filePath, fileInputName, filePart, dummyArchitecture=True, simpleArchitecture=True)
        # Assign inputs to models
        model_inputs = {case: [inputs] for case in cases if 'FULL' in case}
        model_inputs.update({case: [inputs_simple] for case in cases if 'FULL' not in case})
        # MNL models have single input in Input layer
        model_inputs['HYBRID'] = [inputs_lmnl, extra_input_lmnl]
        model_inputs['HYBRID_NAIVE'] = [inputs, extra_input]
        model_inputs['HRUSCHKA'] = [inputs_simple[:,:,i:i+1] for i in range(inputs_simple.shape[2])]
        model_inputs['HRUSCHKA_FULL'] = [inputs[:,:,i:i+1] for i in range(inputs.shape[2])]
        model_inputs['HRUSCH07'] = [inputs_simple[:,:,i:i+1] for i in range(inputs_simple.shape[2])]
        model_inputs['HRUSCH07_FULL'] = [inputs[:,:,i:i+1] for i in range(inputs.shape[2])]
        model_inputs['HYBRID_cor'] = [inputs_lmnl_cor, extra_input_lmnl_cor]
        model_inputs['NN'] = [extra_input_NN]
        model_inputs['SUB'] = [inputs_lmnl_sub, *extra_input_lmnl_sub]
        model_inputs['SUB_corr'] = [inputs_lmnl_subc, *extra_input_lmnl_subc]
        model_inputs['NEST'] = [inputs_lmnl, extra_input_lmnl]
        model_inputs['FULL_NEST'] = [inputs_lmnl_subnl, extra_input_lmnl_subnl]
        model_inputs['LMNL_FULL'] = [inputs, extra_input]
        model_inputs['LMNL_NEST'] = [inputs_lmnl, extra_input_lmnl]
        model_inputs['LMNL'] = [inputs_lmnl, extra_input_lmnl]
        model_inputs['LMNL_FULL_NEST'] = [inputs, extra_input]
        model_inputs['DUMMY_MNL'] = [inputs_Dumy]

        # model_inputs[cases[2]] = [inputs]
        return model_inputs, labels

    model_inputs, train_labels = get_model_inputs_labels('_train')
    model_test_inputs, test_labels = get_model_inputs_labels('_test')
    # Dict to save all values and plotting
    encyclopedia = {}

    for case in cases:
        print(case)
        NN_flag = 'HRUSCHKA' in case
        NN_flag = 'NN' in case
        # Get model extension name and path to .dat file
        path, extend, flag = paths_extends[case]
        model = fetch_model('', path, extend, case)
        betas = 0
        if not NN_flag:
            betas = ghu.get_betas(model)
        likelihood_train, accuracy_train = ghu.get_likelihood_accuracy(model, model_inputs[case], train_labels)
        likelihood_test, accuracy_test = ghu.get_likelihood_accuracy(model, model_test_inputs[case], test_labels)
        gradients = 0
        if 'NEST' not in case:
            gradients = ghu.get_inputs_gradient(model, model_inputs[case], train_labels, flag=flag)
        # Getting STD is slow (1-3s), avoid if not necessary
        stds = 0
        if std_flag and not NN_flag:
            stds = ghu.get_stds(model, model_inputs[case], train_labels)
        encyclopedia['betas_' + case] = betas
        encyclopedia['likelihood_train_' + case] = likelihood_train
        encyclopedia['likelihood_test_' + case] = likelihood_test
        encyclopedia['accuracy_train_' + case] = accuracy_train
        encyclopedia['accuracy_test_' + case] = accuracy_test
        encyclopedia['gradients_' + case] = gradients
        encyclopedia['stds_' + case] = stds

    K.clear_session()
    return encyclopedia


if __name__ == '__main__':
    cases = [
# #        'MNL',
# #        'FULL_MNL',
        # 'HYBRID',
# #        'SUB',
#         'SUB_corr',
#         'NEST'
#         # 'HYBRID_NAIVE',
#         # 'HRUSCH07',
#         # 'HRUSCH07_FULL',
#         # 'HRUSCHKA_FULL',
#         # 'HRUSCHKA',
#         # 'HYBRID_cor',
# #        'NN'
        # 'FULL',
        # 'FULL_NEST',
        # 'LMNL_FULL',
        # 'LMNL_FULL_NEST',
        'LMNL',
        'LMNL_NEST'#,
        # 'DUMMY_MNL'
    ]

    encyclopedia = {}

    paths_extends = {
        'MNL': ['../models/', '_MNL', None],
        'FULL_MNL': ['../models/', '_MNL_Full', None],
        'DUMMY_MNL': ['../models/', '_MNL_Dummy', None],
        'HYBRID': ['../models/', '_Enhancedextra', 'L'],
        'SUB': ['../models/', '_SUB', 'S'],
        'SUB_corr': ['../models/', '_SUB_correl', 'S'],
        'HYBRID_NAIVE': ['../models/', '_Enhanced_Naiveextra', 'L'],
        'HRUSCH07': ['../models/', '_Hruschka07', 'H'],
        'HRUSCH07_FULL': ['../models/', '_Hruschka07_Full', 'H'],
        'HRUSCHKA_FULL': ['../models/', '_Hruschka_Full', 'H'],
        'HRUSCHKA': ['../models/', '_Hruschka', 'H'],
        'HYBRID_cor': ['../models/', '_Enhanced_correlextra', 'L'],
        'NN': ['../models/', '_NN_Fullextra', None],
        'NEST': ['../models/', '_Nest_correl', None],

        'FULL':['../models2/', '_MNL_Full', None],
        'FULL_NEST': ['../models2/', '_Nest_Full', None],
        'LMNL_FULL': ['../models2/', '_Enhanced_Naiveextra', 'L'],
        'LMNL_FULL_NEST': ['../models2/', '_Nest_Naive', None],
        'LMNL': ['../models2/', '_Enhancedextra', 'L'],
#        'LMNL_NEST': ['../models2/', '_Nest_correl', None]
        'LMNL_NEST': ['../models2/', '_Nest', None]
    }

    filePath = '../models/'
    fileInputBase = 'swissmetro'

    encyclopedia = scan_iteration(cases, paths_extends, filePath, fileInputBase, encyclopedia, std_flag=True)

    # pickle.dump(encyclopedia, open('Encyclopedia_VOT.p', 'wb'))

    print('\n\n----- Likelihood Train set -----')
    for case in cases:
        likelihood = encyclopedia['likelihood_train_' + case]
        print(case + ': {}'.format(np.array(likelihood)))

    print('\n\n----- likelihood_test_ -----')
    for case in cases:
        likelihood = encyclopedia['likelihood_test_' + case]
        print(case + ': {}'.format(np.array(likelihood)))

    print('\n\n----- Betas and Stds -----')
    for case in cases:
        betas = encyclopedia['betas_' + case]
        stds = encyclopedia['stds_' + case]
        print('\n' + case + ': {}'.format(np.array(betas)))
        print('stds' + ': {}'.format(np.array(stds)))
        print('t-tests: {}'.format((np.array(betas)/np.array(stds))))

    print('\n\n----- Gradients -----')
    for case in cases:
        gradients = encyclopedia['gradients_' + case]
        print(case + ': {}'.format(np.mean(np.array(gradients), axis=0)))
        print(case + ': {}'.format(np.sum(np.array(gradients), axis=0)))
        print(case + ': {}'.format(np.sum(np.sum(np.array(gradients), axis=0), axis=1)))
