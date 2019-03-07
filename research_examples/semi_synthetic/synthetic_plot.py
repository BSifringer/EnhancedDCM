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
    Prints results for semi-synthetic data models
    
    Main() Flags
    ------------
    illustrative:   models w. semi-synthetic data with low non-linear contribution
    power-log:      models w. semi-synthetic data with high non-linear contribution
"""


if __name__ == '__main__':
    """ Fills cases dictionnary, prints model information """
    cases = [
        'MNL',
        'FULL_MNL',
        'HYBRID',
    ]

    encyclopedia = pickle.load(open('Encyclopedia_.p', 'rb'))
    print('\n\n----- Likelihood Train set -----')
    for case in cases:
        likelihood = encyclopedia['likelihood_train_' + case]
        print(case + ': {}'.format(np.array(likelihood)))

    print('\n\n----- likelihood_test_ -----')
    for case in cases:
        likelihood = encyclopedia['likelihood_test_' + case]
        print(case + ': {}'.format(np.array(likelihood)))

    print('\n\n----- Betas and Stds -----')
    for case in cases[:]:
        betas = encyclopedia['betas_' + case]
        stds = encyclopedia['stds_' + case]
        print('\n' + case + ': {}'.format(np.array(betas)))
        print(case + ': {}'.format(np.array(stds)))
        print('t-tests: {}'.format((np.array(betas)/np.array(stds))))
