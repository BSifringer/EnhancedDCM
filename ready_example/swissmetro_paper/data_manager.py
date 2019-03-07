import numpy as np
import random
import shelve
import _pickle as pickle
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import mean_squared_error
import os


"""
    This script adapts the swissmetro dataset to train on the keras models
"""



fileName = 'swissmetro'


def train_test_split(filePath, seed = None):
    """ Shuffles the dataset and splits in test/train set """

    if seed is not None:
        random.seed(seed)

    lines = open(filePath + fileName + '.dat', 'r').readlines()

    data = np.loadtxt(filePath + fileName + '.dat',skiprows = 1)
    CHOICE = data[:,-1]

    TRAIN = [p for p,i in enumerate(CHOICE) if i == 1]
    SM = [p for p,i in enumerate(CHOICE) if i == 2]
    CAR = [p for p,i in enumerate(CHOICE) if i == 3]

    np.random.shuffle(TRAIN)
    np.random.shuffle(SM)
    np.random.shuffle(CAR)


    test_ratio = 0.8
    step_TRAIN = int(len(TRAIN)*test_ratio)
    step_SM = int(len(SM)*test_ratio)
    step_CAR = int(len(CAR)*test_ratio)

    file_part = open(filePath + fileName + '_train' + '.dat', 'w')
    file_part.writelines(lines[0])
    part = [lines[i+1] for i in TRAIN[:step_TRAIN]]
    file_part.writelines(part)
    part = [lines[i+1] for i in SM[:step_SM]]
    file_part.writelines(part)
    part = [lines[i+1] for i in CAR[:step_CAR]]
    file_part.writelines(part)
    file_part.close()

    file_part = open(filePath + fileName + '_test' + '.dat', 'w')
    file_part.writelines(lines[0])
    part = [lines[i+1] for i in TRAIN[step_TRAIN:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in SM[step_SM:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in CAR[step_CAR:]]
    file_part.writelines(part)
    file_part.close()

    return ['_train', '_test']



def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, lmnlArchitecture = False, write = True):
    """
    Prepares Input for Models. Based on Dataset, utility functions and number of alternatives
    The first input is the X feature set, it ressembles the utility functions.
        - The shape is (n x betas+1 x alternatives), where the added +1 is the label.
    The second input is the Q feature set.
        - The shape is (n x Q_features x 1)
    :param filePath:        path to dataset
    :param fileInputName:   name of dataset
    :param filePart:        dataset extension (e.g. _train, _test)
    :param simpleArchitecture:  Smaller Utility Function
    :param lmnlArchitecture:    L-MNL Utility Function (Small and no ASC)
    :param write:           Save X and Q inputs in a .npy
    :return:    train_data: X inputs Table with Choice label,
                extra_data: Q inputs vector
                train_data_name: saved name to X's .npy file
    """

    extend = ''
    if simpleArchitecture:
        extend = '_simple'
    if lmnlArchitecture:
        extend = '_noASC'

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'

    #filePath = 'swissmetro_paper/'
    filePath = os.path.dirname(os.path.realpath(__file__))+'/'
    data = np.loadtxt(filePath + fileName + filePart + '.dat',skiprows = 1)
    beta_num = 4
    choices_num = 3

    #exclusions:
    CHOICE = data[:,-1]
    PURPOSE = data[:,4]
    CAR_AV = data[:,16]
    TRAIN_AV = data[:,15]
    SM_AV = data[:,17]

    exclude = ((CAR_AV == 0) + (CHOICE == 0) + (TRAIN_AV == 0) + (SM_AV == 0)) > 0
    exclude_list = [i for i, k in enumerate(exclude) if k > 0]

    data = np.delete(data,exclude_list, axis = 0)


    #Define:
    CHOICE = data[:,-1]
    TRAIN_TT = data[:, 18]
    TRAIN_COST = data[:,19] * (data[:,12] == 0) #if he owns a GA
    SM_TT = data[:,21]
    SM_COST = data[:,22] * (data[:,12] == 0) #if he owns a GA
    CAR_TT = data[:,25]
    CAR_CO = data[:,26]

    TRAIN_HE = data[:,20]
    SM_HE = data[:,23]
    GA = data[:,12]
    AGE = data[:,9]

    LUGGAGE = data[:,8]
    SM_SEATS = data[:,24]

    scale = 100.0
    #scale = 1.0

    TRAIN_TT_SCALED = TRAIN_TT/scale
    TRAIN_COST_SCALED = TRAIN_COST/scale
    SM_TT_SCALED = SM_TT/scale
    SM_COST_SCALED = SM_COST/scale
    CAR_TT_SCALED = CAR_TT/scale
    CAR_CO_SCALED = CAR_CO/scale
    TRAIN_HE_SCALED = TRAIN_HE/scale
    SM_HE_SCALED = SM_HE/scale

    ASCs = np.ones(CHOICE.size)
    ZEROs = np.zeros(CHOICE.size)

    CHOICE_CAR = (CHOICE == 3)
    CHOICE_SM = (CHOICE == 2)
    CHOICE_TRAIN = (CHOICE == 1)


    train_data = np.array(
        [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, GA,    AGE,   ZEROs,   ZEROs,    CHOICE_TRAIN],
        [ZEROs,  ASCs,  SM_TT_SCALED,    SM_COST_SCALED,    SM_HE_SCALED,    GA,    ZEROs, ZEROs,   SM_SEATS, CHOICE_SM],
        [ASCs,   ZEROs, CAR_TT_SCALED,   CAR_CO_SCALED,     ZEROs,           ZEROs, ZEROs, LUGGAGE, ZEROs,    CHOICE_CAR]] )

    if simpleArchitecture:    
        train_data = np.array(
            [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
            [ZEROs,  ASCs,  SM_TT_SCALED,    SM_COST_SCALED,    SM_HE_SCALED,    CHOICE_SM],
            [ASCs,   ZEROs, CAR_TT_SCALED,   CAR_CO_SCALED,     ZEROs,    CHOICE_CAR]] )
    if lmnlArchitecture:
        train_data = np.array(
            [[TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
            [SM_TT_SCALED,    SM_COST_SCALED,    SM_HE_SCALED,    CHOICE_SM],
            [CAR_TT_SCALED,   CAR_CO_SCALED,     ZEROs,    CHOICE_CAR]] )
    
    train_data = np.swapaxes(train_data,0,2)

    if simpleArchitecture or lmnlArchitecture:
        # Hybrid Simple
        extra_data = np.delete(data,[18,19,21,22,25,26,27, 20,23, 0,1,2,3, 15, 16,17],axis = 1)
    else:
        # Hybrid MNL
        extra_data = np.delete(data,[18,19,21,22,25,26,27, 20,23, 0,1,2,3, 8, 9, 12, 24, 15, 16,17],axis = 1)


    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name