import numpy as np
import os

""" Semi-Synthetic Dataset -  Data Manager"""

def normalize(data):
    return (data-data.mean(axis=0))/data.std(axis=0)


def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, write = True, lmnlArchitecture = False):
    """
    Prepares Input for Models. Based on Dataset, utility functions and number of alternatives

    :param filePath:        path to dataset
    :param fileInputName:   name of dataset
    :param filePart:        dataset extension (e.g. _train, _test)
    :param simpleArchitecture:  Smaller Utility Function
    :param write:           Save X and Q inputs in a .npy
    :param lmnlArchitecture:    L-MNL Utility Function (Small and no ASC)
    :return:    train_data: X inputs Table with Choice label,
                extra_data: Q inputs vector
                train_data_name: saved name to X's .npy file
    """
    extend = ''
    if simpleArchitecture:
        extend = '_simple'
    if lmnlArchitecture:
        extend = '_noASC'


    train_data_name = filePath + 'keras_input_' + fileInputName + extend + filePart + '.npy'

    filePath = os.path.dirname(__file__)+'/'
    #fileName = 'swissmetro'
    data = np.loadtxt(filePath + fileInputName + filePart + '.dat',skiprows = 1)
    beta_num = 4
    choices_num = 3

    # exclusions:
    CHOICE = data[:, -1]
    PURPOSE = data[:, 4]
    CAR_AV = data[:, 16]
    TRAIN_AV = data[:, 15]
    SM_AV = data[:, 17]

    exclude = ((CAR_AV == 0) + (CHOICE == 0) + (TRAIN_AV == 0) + (SM_AV == 0)) > 0
    exclude_list = [i for i, k in enumerate(exclude) if k > 0]

    data = np.delete(data, exclude_list, axis=0)

    scale = 100.

    # Define:
    CHOICE = data[:, -1]
    TRAIN_TT = data[:, 18]/scale
    TRAIN_COST = data[:, 19] * (data[:, 12] == 0) /scale # if he owns a GA
    SM_TT = data[:, 21] /scale
    SM_COST = data[:, 22] * (data[:, 12] == 0) /scale  # if he owns a GA
    CAR_TT = data[:, 25] /scale
    CAR_CO = data[:, 26] /scale

    TRAIN_HE = data[:, 20] /scale
    SM_HE = data[:, 23] /scale
    GA = data[:, 12]
    ORIGIN = data[:, 13]
    DEST = data[:, 14]
    AGE = data[:, 9]
    INCOME = data[:,10]
    LUGGAGE = data[:, 8]
    SM_SEATS = data[:, 24]

    ASCs = np.ones(CHOICE.size)
    ZEROs = np.zeros(CHOICE.size)

    PURPOSE = data[:, 4]

    x1t, x2t = TRAIN_TT, TRAIN_COST
    x1s, x2s = SM_TT, SM_COST
    x1c, x2c = CAR_TT, CAR_CO

    x3 = GA
    x4 = AGE
    x5 = LUGGAGE
    x6 = PURPOSE
    x7 = SM_SEATS

    x3 = normalize(INCOME)
    x4 = AGE
    x5 = normalize(ORIGIN)
    x6 = normalize(DEST)
    x7 = normalize(PURPOSE)

    """ Utility Specifications """
    train_data = [
        [ZEROs, ZEROs, TRAIN_TT, TRAIN_COST, x6, x4, x5, ZEROs, ZEROs],
        [ZEROs, ASCs, SM_TT, SM_COST, x6, x4, x5, ZEROs, x7],
        [ASCs, ZEROs, CAR_TT, CAR_CO, ZEROs, ZEROs, ZEROs, x3, ZEROs]
    ]

    if simpleArchitecture:
        train_data = [
            [ZEROs, ZEROs, TRAIN_TT, TRAIN_COST],
            [ZEROs, ASCs, SM_TT, SM_COST],
            [ASCs, ZEROs, CAR_TT, CAR_CO]
        ]

    if lmnlArchitecture:
        train_data = [
            [TRAIN_TT, TRAIN_COST],
            [SM_TT, SM_COST],
            [CAR_TT, CAR_CO]
        ]


    if simpleArchitecture or lmnlArchitecture:
        # Hybrid Simple
        extra_data = np.delete(data,[18,19,21,22,25,26,27, 0,1,2,3, 15, 16, 17],axis = 1)
    else:
        # Hybrid MNL
        extra_data = np.delete(data,[18,19,21,22,25,26,27, 0,1,2,3, 8,9,12,24,4, 15,16,17],axis = 1)

    train_data = np.swapaxes(train_data,0,2)

    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name
