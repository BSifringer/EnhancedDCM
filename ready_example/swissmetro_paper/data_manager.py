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

def normalize(data):
    return (data-data.mean(axis=0))/(data.std(axis=0))

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



def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, lmnlArchitecture = False,
        correlArchitecture = False, nnArchitecture = False, subArchitecture=False, nestArchitecture=False,
        nlArchitecture=False, dummyArchitecture=False, write = True):
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
    if correlArchitecture:
        extend = '_correl'
    if subArchitecture:
        extend = '_sub'
    if nlArchitecture:
        extend= '_nl'

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

#    TRAIN_TT_SCALED = normalize(TRAIN_TT)
#    SM_TT_SCALED = normalize(SM_TT)
#    CAR_TT_SCALED = normalize(CAR_TT)

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
    if correlArchitecture or subArchitecture:
        if not lmnlArchitecture :
            train_data = np.array(
                [[ TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
                [    SM_COST_SCALED,    SM_HE_SCALED,    CHOICE_SM],
                [   CAR_CO_SCALED,     ZEROs,    CHOICE_CAR]] )
                # [[ TRAIN_HE_SCALED, CHOICE_TRAIN],
                # [       SM_HE_SCALED,    CHOICE_SM],
                # [       ZEROs,    CHOICE_CAR]] )

    if nestArchitecture:
        train_data = np.array(
            [[TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, ZEROs, ZEROs, ZEROs,     ZEROs, ZEROs,    CHOICE_TRAIN],
            [ZEROs, ZEROs, ZEROs,         SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED,        ZEROs, ZEROs,    CHOICE_SM],
            [ZEROs,ZEROs, ZEROs,             ZEROs, ZEROs, ZEROs,        CAR_TT_SCALED,   CAR_CO_SCALED,     CHOICE_CAR]] )

    if simpleArchitecture or lmnlArchitecture or correlArchitecture or subArchitecture or nestArchitecture:
        # Hybrid Simple
        extra_data = np.delete(data,[18,19,21,22,25,26,27, 20,23, 0,1,2,3, 15, 16,17],axis = 1)
        if correlArchitecture and not subArchitecture:
            costs = np.array([TRAIN_TT_SCALED, SM_TT_SCALED, CAR_TT_SCALED])
            costs = np.swapaxes(costs,0,1)
            extra_data = np.concatenate((extra_data, costs), axis=1)
        if subArchitecture:
            # sub1 = np.concatenate((extra_data, np.expand_dims(TRAIN_TT_SCALED, axis=-1)), axis=1)
            # sub2 = np.concatenate((extra_data, np.expand_dims(SM_TT_SCALED, axis=-1)), axis=1)
            # sub3 = np.concatenate((extra_data, np.expand_dims(CAR_TT_SCALED, axis=-1)), axis=1)
            # sub1 = np.concatenate((extra_data, np.expand_dims(TRAIN_TT_SCALED, axis=-1),np.expand_dims(TRAIN_COST_SCALED, axis=-1)), axis=1)
            # sub2 = np.concatenate((extra_data, np.expand_dims(SM_TT_SCALED, axis=-1),np.expand_dims(SM_COST_SCALED, axis=-1)), axis=1)
            # sub3 = np.concatenate((extra_data, np.expand_dims(CAR_TT_SCALED, axis=-1),np.expand_dims(CAR_CO_SCALED, axis=-1)), axis=1)
            # extra_data = np.array([sub1, sub2, sub3])
            extra_data = np.array([extra_data, extra_data, extra_data])

    elif nnArchitecture:
        extra_data = np.delete(data, [0,1,2,3, 27], axis=1)
    elif nlArchitecture:
        extra_data = np.array([ZEROs])
        extra_data = np.swapaxes(extra_data,0,1)
    else:
        # Hybrid MNL
        extra_data = np.delete(data,[18,19,21,22,25,26,27, 20,23, 0,1,2,3, 8, 9, 12, 24, 15, 16,17],axis = 1)


    if dummyArchitecture:
        extra_data = np.swapaxes(extra_data,0,1)
        train_data = np.array(train_data)
        print(train_data.shape)
        labels = train_data[:,-1:,:]
        train_data = np.delete(train_data, -1, axis = 1)
        extra_range = extra_data.shape[0]
        ranged_Zero = np.expand_dims(np.array([ZEROs for i in range(extra_range)]), axis=0)
        extra_data_dum = np.expand_dims(extra_data, axis=0)
        print(train_data.shape)
        print(ranged_Zero.shape)
        print(extra_data_dum.shape)
        print(np.concatenate([extra_data_dum, ranged_Zero, ranged_Zero]).shape)
        for j in range(choices_num):
            alternative_dummy = np.concatenate([ranged_Zero if i!=j else extra_data_dum for i in range(choices_num)])
            train_data = np.concatenate((train_data, alternative_dummy), axis=1)
        print(train_data.shape)
        train_data = np.concatenate((train_data, labels), axis=1)


    train_data = np.swapaxes(train_data,0,2)

    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    print(CHOICE_TRAIN.sum())
    print(CHOICE_SM.sum())
    print(CHOICE_CAR.sum())

    return train_data, extra_data, train_data_name
