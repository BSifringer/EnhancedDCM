import numpy as np
from keras.models import Sequential, Model
from keras.models import load_model
import pandas as pd
import os

""" Synthetic Dataset -  Data Manager"""

fileName = 'guevara'
data_folder = os.path.dirname(os.path.realpath(__file__))+'/TRA_2015/'
#data_folder = os.path.dirname(os.path.realpath(__file__))+'/TRA_log/'
# data_folder = os.path.dirname(os.path.realpath(__file__))+'/TRA_exp/'
# data_folder = os.path.dirname(os.path.realpath(__file__))+'/TRA_interact/'

def normalize(data):
    return (data-data.mean(axis=0))/(data.std(axis=0))


def keras_input_scan(filePath, fileInputName, iteration, utility_indices = [], filePart = '', write = False, extend='',
    latentArchitecture = False, trueArchitecture = False, layerArchitecture = False, weakArchitecture = False,
    multipleArchitecture = False, multipleWeakArchitecture = False, correlArchitecture = False):

    dp1 = pd.read_csv(data_folder + "dp1.csv")
    dp2 = pd.read_csv(data_folder + "dp2.csv")
    da1 = pd.read_csv(data_folder + "da1.csv")
    da2 = pd.read_csv(data_folder + "da2.csv")
    db1 = pd.read_csv(data_folder + "db1.csv")
    db2 = pd.read_csv(data_folder + "db2.csv")
    di1 = pd.read_csv(data_folder + "di1.csv")
    di2 = pd.read_csv(data_folder + "di2.csv")
    dh1 = pd.read_csv(data_folder + "dh1.csv")
    dh2 = pd.read_csv(data_folder + "dh2.csv")
    dk1 = pd.read_csv(data_folder + "dk1.csv")
    dk2 = pd.read_csv(data_folder + "dk2.csv")
    dCh1 = pd.read_csv(data_folder + "dCh1.csv")
    dCh2 = pd.read_csv(data_folder + "dCh2.csv")
    # dCh1 = pd.read_csv(data_folder + "dCh1log.csv")
    # dCh2 = pd.read_csv(data_folder + "dCh2log.csv")
    dq1 = pd.read_csv(data_folder + "dq1.csv")
    dq2 = pd.read_csv(data_folder + "dq2.csv")
    # dq3 = pd.read_csv(data_folder + "dq3.csv")
    # dq31 = pd.read_csv(data_folder + "dq31.csv")
    # dq32 = pd.read_csv(data_folder + "dq32.csv")

    #dCh1log = pd.read_csv(data_folder + "dCh1log.csv")
    #  dCh2log = pd.read_csv(data_folder + "dCh2log.csv")

    dmis11 = pd.read_csv(data_folder + "dmis11.csv")
    dmis12 = pd.read_csv(data_folder + "dmis12.csv")
    dmis21 = pd.read_csv(data_folder + "dmis21.csv")
    dmis22 = pd.read_csv(data_folder + "dmis22.csv")
    # dmis13 = pd.read_csv(data_folder + "dmis13.csv")
    # dmis23 = pd.read_csv(data_folder + "dmis23.csv")

    dqmis11 = pd.read_csv(data_folder + "dqmis11.csv")
    dqmis12 = pd.read_csv(data_folder + "dqmis12.csv")
    dqmis21 = pd.read_csv(data_folder + "dqmis21.csv")
    dqmis22 = pd.read_csv(data_folder + "dqmis22.csv")

    dwmis11 = pd.read_csv(data_folder + "dwmis11.csv")
    dwmis12 = pd.read_csv(data_folder + "dwmis12.csv")
    dwmis21 = pd.read_csv(data_folder + "dwmis21.csv")
    dwmis22 = pd.read_csv(data_folder + "dwmis22.csv")


    dz1 = pd.read_csv(data_folder + "dz1.csv")
    dz2 = pd.read_csv(data_folder + "dz2.csv")

    dwz1 = pd.read_csv(data_folder + "dwz1.csv")
    dwz2 = pd.read_csv(data_folder + "dwz2.csv")

    dxz1 = pd.read_csv(data_folder + "dxz1.csv")
    dxz2 = pd.read_csv(data_folder + "dxz2.csv")

    dxi1 = pd.read_csv(data_folder + "dxi1.csv")
    dxi2 = pd.read_csv(data_folder + "dxi2.csv")



    p1 = dp1.iloc[:,iteration].values
    p2 = dp2.iloc[:,iteration].values
    a1 = da1.iloc[:,iteration].values
    a2 = da2.iloc[:,iteration].values
    b1 = db1.iloc[:,iteration].values
    b2 = db2.iloc[:,iteration].values
    i1 = di1.iloc[:,iteration].values
    i2 = di2.iloc[:,iteration].values
    h1 = dh1.iloc[:,iteration].values
    h2 = dh2.iloc[:,iteration].values
    k1 = dk1.iloc[:,iteration].values
    k2 = dk2.iloc[:,iteration].values
    Ch1 = dCh1.iloc[:,iteration].values
    Ch2 = dCh2.iloc[:,iteration].values
    q1 = dq1.iloc[:,iteration].values
    q2 = dq2.iloc[:,iteration].values
    # q3 = dq3.iloc[:,iteration].values
    # q31 = dq31.iloc[:,iteration].values
    # q32 = dq32.iloc[:,iteration].values


    mis11 = dmis11.iloc[:,iteration].values
    mis12 = dmis12.iloc[:,iteration].values
    mis21 = dmis21.iloc[:,iteration].values
    mis22 = dmis22.iloc[:,iteration].values

    # mis13 = dmis21.iloc[:,iteration].values
    # mis23 = dmis22.iloc[:,iteration].values

    qmis11 = dqmis11.iloc[:,iteration].values
    qmis12 = dqmis12.iloc[:,iteration].values
    qmis21 = dqmis21.iloc[:,iteration].values
    qmis22 = dqmis22.iloc[:,iteration].values

    wmis11 = dwmis11.iloc[:,iteration].values
    wmis12 = dwmis12.iloc[:,iteration].values
    wmis21 = dwmis21.iloc[:,iteration].values
    wmis22 = dwmis22.iloc[:,iteration].values

    z1 = dz1.iloc[:,iteration].values
    z2 = dz2.iloc[:,iteration].values


    wz1 = dwz1.iloc[:,iteration].values
    wz2 = dwz2.iloc[:,iteration].values


    xz1 = dxz1.iloc[:,iteration].values
    xz2 = dxz2.iloc[:,iteration].values


    xi1 = dxi1.iloc[:,iteration].values
    xi2 = dxi2.iloc[:,iteration].values

    ASCs = np.ones(p1.size)
    ZEROs = np.zeros(p1.size)
    # print(np.concatenate([p1,p2]).shape)
    # pn = normalize( np.concatenate([p1,p2]))
    # an = normalize(np.concatenate([a1,a2]))
    # bn = normalize(np.concatenate([b1,b2]))

    #Endogeneous
    train_data = np.array([
        [ZEROs, p1, a1, b1, Ch1],
        [ASCs,  p2, a2, b2, Ch2]
        ])
    # train_data = np.array([
    #     [ZEROs, pn[:1000], an[:1000], bn[:1000], Ch1],
    #     [ASCs,  pn[1000:], an[1000:], bn[1000:], Ch2]
    #     ])

    if latentArchitecture:
        train_data = np.array([
            [ZEROs, p1, a1, b1, mis11, mis21, Ch1],
            [ASCs,  p2, a2, b2, mis12, mis22, Ch2]
            ])
        # train_data = np.array([
        #     [ZEROs, p1, a1, b1, mis11, mis21, mis13, mis23, Ch1],
        #     [ASCs,  p2, a2, b2, mis12, mis22, mis13, mis23, Ch2]
        #     ])
    #True:
    if trueArchitecture:
        train_data = np.array([
            [ZEROs, p1, a1, b1, q1, Ch1],
            [ASCs,  p2, a2, b2, q2, Ch2]
            ])


    extra_data = np.array([mis11, mis12, mis21, mis22])
    # extra_data = np.array([mis11, mis12, mis21, mis22, mis13, mis23])

    if layerArchitecture:
        extra_data =  np.array([mis11, mis12, mis21, mis22, p1, a1, b1, p2, a2, b2])

    if weakArchitecture:
        extra_data =  np.array([wmis11, wmis12, wmis21, wmis22])
        train_data = np.array([
            [ZEROs, p1, a1, b1, wmis11, wmis21, Ch1],
            [ASCs,  p2, a2, b2, wmis12, wmis22, Ch2]
            ])
    if multipleArchitecture:
        extra_data = np.array([mis11, mis12, mis21, mis22, h1, h2])

    if multipleWeakArchitecture:
        extra_data = np.array([wmis11, wmis12, wmis21, wmis22, h1])

    if correlArchitecture:
        extra_data = np.array([q1,q2])
        # extra_data = np.array([q31, q32, q3])

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'

    train_data = np.swapaxes(train_data,0,2)
    extra_data = np.swapaxes(extra_data,0,1)


    if write:
        np.save(train_data_name, train_data)
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

   # train_data[:,:-1,:] = normalize(train_data[:,:-1,:])

    return train_data, extra_data, train_data_name, len(train_data[0])-1, extra_data[0].size

    """
    xls = pd.ExcelFile('~/Desktop/SLTDATABASEBrian.xlsx')
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    n_entries = len(df.ID.values)
    n_people = df.ID.values[-1]

    #print("Success is {} out of {} for full set (ra: {})".format(choice.sum(), len(choice), choice.sum()/len(choice)))
    if (filePart in ['_train', '_test']):
        ratio = 0.9
        share = int(ratio * n_people)
        np.random.seed(1)
        indices = np.arange(n_people)
        np.random.shuffle(indices)
        train_ind = indices[:share]
        test_ind = indices[share:]

        df_train, df_test = [], []
        for i in range(n_entries):
            if df.ID.values[i] in test_ind:
                df_test.append(i)
            else:
                df_train.append(i)

        if filePart == '_train':
            df = df.drop(df_test)
        else:
            df = df.drop(df_train)

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'


    exclude = [0,1,2,4,18,19,20]
    skip = 0
    betas = []
    for i,keep in enumerate(utility_indices):
        while i+skip in exclude:
            skip += 1
        if keep:
            betas.append(i+skip)


    choice = df.QS_1.values>=success
    choice1 = choice == 1

    #print("Success is {} out of {} for  subset (ra: {})".format(choice.sum(), len(choice), choice.sum()/len(choice)))


    train_data = np.array(df.iloc[:,list(betas)])
    print(train_data.shape)
    train_data = np.append(train_data, np.expand_dims(choice1, axis=-1), axis=-1 )
    train_data = np.expand_dims(train_data, axis=-1)

    col_mean = np.nanmean(train_data, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(train_data))
    #Place column means in the indices. Align the arrays using take
    train_data[inds] = np.take(col_mean, inds[1])

    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))


    columns_used = betas + exclude
    extra_columns = np.setdiff1d(np.arange(2,20), np.array(columns_used))

    extra_data = np.array(df.iloc[:,list(extra_columns)])

    col_mean = np.nanmean(extra_data, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(extra_data))
    #Place column means in the indices. Align the arrays using take
    extra_data[inds] = np.take(col_mean, inds[1])

    if write:
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name, len(betas), len(extra_columns)
    """



def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, write = True, lmnlArchitecture = False, NNArchitecture=False):
    success = 2

    """
    Prepares Input for Models. Based on Dataset, utility functions and number of alternatives

    :param filePath:        path to dataset
    :param fileInputName:   name of dataset
    :param filePart:        dataset extension (e.g. _train, _test)
    :param simpleArchitecture:  Smaller Utility Function
    :param write:           Save X and Q inputs in a .npy
    :param lmnlArchitecture:    L-MNL Utility Function (Small and no ASC)
    :param NNArchitecture:    Ground Truth Utility Specification
    :return:    train_data: X inputs Table with Choice label,
                extra_data: Q inputs vector
                train_data_name: saved name to X's .npy file
    """


    xls = pd.ExcelFile('~/Desktop/SLTDATABASEBrian.xlsx')
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    n_entries = len(df.ID.values)
    n_people = df.ID.values[-1]


    if (filePart in ['_train', '_test']):
        ratio = 0.9
        share = int(ratio * n_people)
        np.random.seed(1)
        indices = np.arange(n_people)
        np.random.shuffle(indices)
        train_ind = indices[:share]
        test_ind = indices[share:]

        df_train, df_test = [], []
        for i in range(n_entries):
            if df.ID.values[i] in test_ind:
                df_test.append(i)
            else:
                df_train.append(i)

        if filePart == '_train':
            df = df.drop(df_test)
        else:
            df = df.drop(df_train)


    extend = ''

    if simpleArchitecture:
        extend = '_simple'
    if lmnlArchitecture:
        extend = '_noASC'
    if NNArchitecture:
        extend = '_NN'

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'

    choices_num = 1


    #Define:
    x1 = df.sex.values
    x2 = df.PreRNFL.values/10
    x3 = df.CCT.values/100
    x4 = df.PreMD.values
    x5 = df.PresLV.values
    x6 = df.PreIOP.values
    choice = df.QS_1.values>=success
    print('Succes {} count: {}'.format(success,choice.sum()))


    exclude = ['DOB', 'SLT_1date']
    betas = ['sex', 'PreRNFL', 'CCT', 'PreMD', 'PresLV', 'PreIOP']
    betas = ['sex', 'CCT', 'PreMD', 'PreIOP']
    #betas_final = [betas for beta, keep in zip(betas, [True,False,True,True,False,True]) if keep]

    if lmnlArchitecture:
        betas = betas[:3]
    if NNArchitecture:
        betas = []

    betas = betas + exclude
    columns_used = []
    for beta in betas:
        columns_used.append(np.where(df.columns.values==beta)[0][0])

    extra_columns = np.setdiff1d(np.arange(2,25), np.array(columns_used))


    choice1 = choice == 1
    #choice2 = choice == 0

    ASCs = np.ones(choice.size)
    ZEROs = np.zeros(choice.size)

    """Utility Specifications: """

    train_data = np.array(
#        [[x1,  x2, x3, x4, x5, x6, choice1]#,
        [[x1, x3, x4, x6, choice1]#,
      #   [ZEROs, ZEROs, x2, ZEROs, ZEROs, ZEROs, choice2]
          ])

    if simpleArchitecture:
        train_data = np.array(
            [[ASCs, x1, x2, choice1]])

    if lmnlArchitecture:
        train_data = np.array(
            [[x1, x2, x3, choice1]#,
          #  [ZEROs, x2, choice1]
        ])

    if NNArchitecture:
        train_data = np.array(
            [[choice1]])

    train_data = np.swapaxes(train_data,0,2)


    col_mean = np.nanmean(train_data, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(train_data))
    #Place column means in the indices. Align the arrays using take
    train_data[inds] = np.take(col_mean, inds[1])



    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))


    # if simpleArchitecture or lmnlArchitecture:
    #     # Hybrid Simple
    #     extra_data = np.delete(data,delete_list,axis = 1)
    # else:
    #     # Hybrid MNL
    #     extra_data = np.delete(data,range(len(data)),axis = 1)

    extra_data = np.array(df.iloc[:,list(extra_columns)])

    col_mean = np.nanmean(extra_data, axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(extra_data))
    #Place column means in the indices. Align the arrays using take
    extra_data[inds] = np.take(col_mean, inds[1])

    if write:
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name


if __name__ == '__main__':
    keras_input('','medical','')
