import numpy as np
from keras.models import Sequential, Model
from keras.models import load_model

""" Synthetic Dataset -  Data Manager"""

fileName = 'generated'

def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, write = True, lmnlArchitecture = False, trueArchitecture=False, correlArchitecture=False, subArchitecture=False):
    """
    Prepares Input for Models. Based on Dataset, utility functions and number of alternatives

    :param filePath:        path to dataset
    :param fileInputName:   name of dataset
    :param filePart:        dataset extension (e.g. _train, _test)
    :param simpleArchitecture:  Smaller Utility Function
    :param write:           Save X and Q inputs in a .npy
    :param lmnlArchitecture:    L-MNL Utility Function (Small and no ASC)
    :param trueArchitecture:    Ground Truth Utility Specification
    :return:    train_data: X inputs Table with Choice label,
                extra_data: Q inputs vector
                train_data_name: saved name to X's .npy file
    """
    extend = ''
    if simpleArchitecture:
        extend = '_simple'
    if lmnlArchitecture:
        extend = '_noASC'
    if trueArchitecture:
        extend = '_true'
    if correlArchitecture:
        extend = '_correl'
    if subArchitecture:
        extend = '_sub'

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'

    data = np.loadtxt(filePath + fileInputName + filePart + '.dat',skiprows = 1)
    choices_num = 2

    # print(data.shape)
    #Define:
    p1 = data[:,0]
    p2 = data[:,1]
    a1 = data[:,2]
    a2 = data[:,3]
    b1 = data[:,4]
    b2 = data[:,5]
    q1 = data[:,6]
    q2 = data[:,7]
    c1 = data[:,8]
    c2 = data[:,9]
    choice = data[:,-1]

    choice1 = choice == 1
    choice2 = choice == 0

    ASCs = np.ones(choice.size)
    ZEROs = np.zeros(choice.size)

    """Utility Specifications: """
    train_data = np.array(
        [[ASCs, p1,  a1, b1, q1, c1, choice1],
         [ZEROs, p2, a2, b2, q2, c2, choice2]
          ])

    if simpleArchitecture:
        train_data = np.array(
            [[ASCs, p1, a1, b1, choice1],
            [ZEROs, p2, a2, b2, choice2]
            ])

    if lmnlArchitecture or correlArchitecture:
        train_data = np.array(
            [[p1, a1, b1, choice1],
            [p2, a2, b2, choice2]
        ])

    if trueArchitecture:
        train_data = np.array(
            [[p1, a1, b1, q1*c1, choice1],
            [p2, a2, b2, q2*c2, choice2]
            ])
    if subArchitecture:
        train_data = np.array(
            [[ASCs, ZEROs, choice1],
             [ASCs, ZEROs, choice2]
             ])

    train_data = np.swapaxes(train_data,0,2)

    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))

    delete_list = range(len(data[0]))
    # print(delete_list)
    delete_list = np.delete(delete_list, [6,7,8,9])
    # print(delete_list)

    if simpleArchitecture or lmnlArchitecture:
        # Hybrid Simple
        extra_data = np.delete(data,delete_list,axis = 1)
    elif correlArchitecture:
        extra_data = data[:,:9]
    elif subArchitecture:
        extra_data = [data[:,:9]]
    else:
        # Hybrid MNL
        extra_data = np.delete(data,range(len(data)),axis = 1)

    if write:
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)
    # print(extra_data[0].size)
    return train_data, extra_data, train_data_name, len(train_data[0])-1, extra_data[0].size
