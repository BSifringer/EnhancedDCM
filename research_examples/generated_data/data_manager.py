import numpy as np
from keras.models import Sequential, Model
from keras.models import load_model

""" Synthetic Dataset -  Data Manager"""

fileName = 'generated'

def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, write = True, lmnlArchitecture = False, trueArchitecture=False):
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

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'

    data = np.loadtxt(filePath + fileInputName + filePart + '.dat',skiprows = 1)
    choices_num = 2


    #Define:
    x1 = data[:,0]
    x2 = data[:,1]
    x3 = data[:,2]
    x4 = data[:,3]
    x5 = data[:,4]
    x6 = data[:,5]
    choice = data[:,-1]

    choice1 = choice == 1
    choice2 = choice == 0

    ASCs = np.ones(choice.size)
    ZEROs = np.zeros(choice.size)

    """Utility Specifications: """
    train_data = np.array(
        [[ASCs, x1,  x2, x3, x4, x5, choice1]#,
      #   [ZEROs, ZEROs, x2, ZEROs, ZEROs, ZEROs, choice2]
          ])

    if simpleArchitecture:    
        train_data = np.array(
            [[ASCs, x1, x2, choice1]])

    if lmnlArchitecture:
        train_data = np.array(
            [[x1, x2, choice1]#,
          #  [ZEROs, x2, choice1]
        ])

    if trueArchitecture:
        train_data = np.array(
            [[x1, x2, x3*x4, x3*x5, choice1]])

    train_data = np.swapaxes(train_data,0,2)

    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))

    delete_list = range(len(data))
    delete_list = np.delete(delete_list, [2,3,4])

    if simpleArchitecture or lmnlArchitecture:
        # Hybrid Simple
        extra_data = np.delete(data,delete_list,axis = 1)
    else:
        # Hybrid MNL
        extra_data = np.delete(data,range(len(data)),axis = 1)

    if write:
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name
