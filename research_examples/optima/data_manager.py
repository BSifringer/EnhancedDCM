import numpy as np
import random
#import matplotlib.pyplot as plt
import shelve
import _pickle as pickle
#from pandas import read_csv, get_dummies
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Add, Reshape
from keras.optimizers import RMSprop, Adam, SGD #, multiAdam
from keras.losses import mean_squared_error
import os
#import matplotlib.pyplot as plt
import pandas as pd
fileName = 'optima'


def train_test_split(seed = 1):
    # STRC seed = 1
    # Have to change because test set not optimized for features dummy

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    data_folder = os.path.dirname(os.path.realpath(__file__))
    data_folder = data_folder +'/biogeme/'
    filePath = data_folder
    lines = open(filePath + fileName + '.dat', 'r').readlines()

    data = np.loadtxt(filePath + fileName + '.dat',skiprows = 1)
    CHOICE = data[:,-7]

    PT = [p for p,i in enumerate(CHOICE) if i == 0]
    CAR = [p for p,i in enumerate(CHOICE) if i == 1]
    SM = [p for p,i in enumerate(CHOICE) if i == 2]

    np.random.shuffle(PT)
    np.random.shuffle(SM)
    np.random.shuffle(CAR)


    test_ratio = 0.8
    step_PT = int(len(PT)*test_ratio)
    step_SM = int(len(SM)*test_ratio)
    step_CAR = int(len(CAR)*test_ratio)

    file_part = open(filePath + fileName + '_train' + '.dat', 'w')
    file_part.writelines(lines[0])
    part = [lines[i+1] for i in PT[:step_PT]]
    file_part.writelines(part)
    part = [lines[i+1] for i in SM[:step_SM]]
    file_part.writelines(part)
    part = [lines[i+1] for i in CAR[:step_CAR]]
    file_part.writelines(part)
    file_part.close()

    file_part = open(filePath + fileName + '_test' + '.dat', 'w')
    file_part.writelines(lines[0])
    part = [lines[i+1] for i in PT[step_PT:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in SM[step_SM:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in CAR[step_CAR:]]
    file_part.writelines(part)
    file_part.close()



def keras_input(filePath, fileInputName, filePart = '', write=True, fullArchitecture=False,
    dummyArchitecture = False, simpleArchitecture=False, NNArchitecture=False, inArchitecture=False):

    extend = ''
    if fullArchitecture:
        extend = '_full'

    data_folder = os.path.dirname(os.path.realpath(__file__))
    data_folder = data_folder +'/biogeme/'
    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'
    data = np.loadtxt( data_folder + fileName + filePart + '.dat',skiprows = 1)
    df = pd.read_csv( data_folder + fileName + filePart + '.dat', sep ='\t')
    beta_num = 8
    choices_num = 3

    def list_exclude(name, df, exclude_list):
        column = df[name].values
        exclude = (column == -1)
        exclude_list.extend([i for i,k in enumerate(exclude) if k > 0])
        # df = df.drop(exclude_list)
        return exclude_list

    def pre_exclude(name_list, df):
        exclude_list = []
        for name in name_list:
            exclude_list = list_exclude(name, df, exclude_list)
        df = df.drop(exclude_list)
        return df


        # exclude = (column == -1)
        # exclude_list.extend([i for i,k in enumerate(exclude) if k > 0])
        # # df = df.drop(exclude_list)
        # return exclude_list

    variables = ['Choice', 'TimePT', 'TimeCar', 'CostPT', 'distance_km', 'CostCarCHF',
    'TripPurpose', 'age', 'NbBicy', 'NbCar', 'NbChild', 'HouseType', 'Gender', 'GenAbST',
     'Income', 'Education', 'SocioProfCat', 'OwnHouse', 'FamilSitu', 'CalculatedIncome']

    # variables = ['Choice', 'TimePT', 'TimeCar', 'CostPT', 'distance_km', 'CostCarCHF',
    #  'TripPurpose', 'age', 'NbBicy', 'NbCar', 'NbChild',
    #  'Income']

    #exclusions:
    # CHOICE = data[:,-7]
    df = pre_exclude(variables, df)

    def fetch(name, df=df):
        return df[name].values
    CHOICE = fetch('Choice')
#    print(CHOICE.size)
    #PURPOSE = data[:,4]
    # exclude = (CHOICE == -1)
    # exclude_list = [i for i,k in enumerate(exclude) if k > 0]

    # data = np.delete(data,exclude_list, axis = 0)
    # df = df.drop(exclude_list)

#    print(df.columns)
    #Define:
    # CHOICE = data[:,-7]
    # TimePT = data[:, 3]
    # PT_COST = data[:,6]
    # distance_km = data[:,-8]
    # #Gender = data[:,26]
    # TimeCar = data[:,8]
    # CostCarCHF = data[:,7]
    # TripPurpose = data[:,-21]

    TimePT = fetch('TimePT')
    PT_COST = fetch('CostPT')
    distance_km = fetch('distance_km')
    TimeCar = fetch('TimeCar')
    CostCarCHF = fetch('CostCarCHF')
    TripPurpose = fetch('TripPurpose')

    MarginalCostPT = np.array(df['MarginalCostPT'].values)
    WaitingTimePT = np.array(df['WaitingTimePT'].values)
    #GENDER = [p if p == 1 else 0 for p in Gender]

    ASCs = np.ones(CHOICE.size)
    ZEROs = np.zeros(CHOICE.size)

    CHOICE_CAR = (CHOICE == 1)
    CHOICE_WALK = (CHOICE == 2)
    CHOICE_PT = (CHOICE == 0)
    TimePT_scaled  = TimePT/200
    TimeCar_scaled  = TimeCar/200
    MarginalCostPT_scaled  = MarginalCostPT/10
    CostCarCHF_scaled  = CostCarCHF/10
    distance_km_scaled  = distance_km/5
    PurpHWH = TripPurpose == 1
    PurpOther = TripPurpose != 1


    # age = data[:,-2]
    # NbCar = data[:,11]
    # NbBicy = data[:,13]
    # HouseType = data[:,21]

    age = fetch('age')
    NbCar = fetch('NbCar')
    NbBicy = fetch('NbBicy')
    HouseType = fetch('HouseType')

    Gender = df['Gender'].values
    GenAbST = df['GenAbST'].values
    Education = df['Education'].values
    # FamilSitu = data[:,29]
    FamilSitu = fetch('FamilSitu')

    CalculatedIncome = df['CalculatedIncome'].values
    Income = df['Income'].values
    NbChild = df['NbChild'].values


    Gender = fetch('Gender')
    GenAbST = fetch('GenAbST')
    Education = fetch('Education')
    CalculatedIncome = fetch('CalculatedIncome')
    Income = fetch('Income')
    NbChild = fetch('NbChild')
    NbTransf = fetch('NbTransf')
    DestAct = fetch('DestAct')
    TypeCommune = fetch('TypeCommune')
    FreqTripHouse = fetch('FreqTripHouseh')
    ModeToSchool = fetch('ModeToSchool')
    NbTrajects = fetch('NbTrajects')

    OwnHouse = fetch('OwnHouse')
    Mothertongue = fetch('Mothertongue')
    SocioProfCat = fetch('SocioProfCat')
#    print(SocioProfCat.size)
    def get_binaries(Variable):
        Variable_cat = pd.get_dummies(Variable)
        Variable_label_names = Variable_cat.columns
        Variable_binaries = np.array(Variable_cat.as_matrix())
        if Variable_label_names[0] == -1:
            Variable_binaries = Variable_binaries[:,1:]
        print(Variable_binaries[0].size)
        return Variable_binaries

    ModeToSchool_binaries = get_binaries(ModeToSchool)
    HouseType_binaries = get_binaries(HouseType)
    DestAct_binaries = get_binaries(DestAct)
    SocioProfCat_binaries = get_binaries(SocioProfCat)
    Education_binaries = get_binaries(Education)
    FamilSitu_binaries = get_binaries(FamilSitu)
    TripPurpose_binaries = get_binaries(TripPurpose)

    French = df['LangCode'].values == 1
    Student = df['OccupStat'].values==8
    Urban = df['UrbRur'].values == 2
    ScaledIncome = CalculatedIncome/1000
    Work = TripPurpose==1

    age_65_more = age >=65
    age_25_less = age <=25

### DEFINITION OF UTILITY FUNCTIONS:
    """
    BETA_TIME_PT = BETA_TIME_PT_REF * \
    exp(BETA_TIME_PT_CL * CARLOVERS)
    """
    # train_data = np.array(
    #     [[ZEROs,ZEROs, TimePT_scaled,  MarginalCostPT_scaled*PurpHWH, MarginalCostPT_scaled*PurpOther, WaitingTimePT, ZEROs,              CHOICE_PT],
    #     [ASCs,  ZEROs, TimeCar_scaled, CostCarCHF_scaled*PurpHWH,   CostCarCHF_scaled*PurpOther,      ZEROs,         ZEROs,              CHOICE_CAR],
    #     [ZEROs, ASCs,  ZEROs,          ZEROs,                         ZEROs,                         ZEROs,         distance_km_scaled, CHOICE_WALK]] )
    #
    #
    # extra_data = np.array([age_65_more,NbCar,NbBicy,HouseType,Gender, GenAbST,Education,FamilSitu,ScaledIncome])

    train_data = np.array(
        [[ZEROs,ZEROs, TimePT_scaled,  ZEROs, MarginalCostPT_scaled/ScaledIncome, ZEROs,   ZEROs, ZEROs, ZEROs, Student,Urban, ZEROs, ZEROs, CHOICE_PT],
        [ASCs,  ZEROs, ZEROs, TimeCar_scaled, CostCarCHF_scaled/ScaledIncome,     NbChild, NbCar, Work, French,ZEROs,  ZEROs, ZEROs, ZEROs, CHOICE_CAR],
        [ZEROs, ASCs,  ZEROs, ZEROs,          ZEROs,                              ZEROs,   ZEROs,  ZEROs,ZEROs, ZEROs,  ZEROs, distance_km_scaled, NbBicy, CHOICE_WALK]] )


    extra_data = np.array([age, HouseType,Gender, GenAbST,Education,FamilSitu,ScaledIncome, OwnHouse, Mothertongue, SocioProfCat])

    if dummyArchitecture:
            train_data = np.delete(train_data, -1, axis = 1)
            extra_range = extra_data.shape[0]
            ranged_Zero = np.expand_dims(np.array([ZEROs for i in range(extra_range)]), axis=0)
            extra_data = np.expand_dims(extra_data, axis=0)

            pt_dummy = np.concatenate((extra_data, ranged_Zero, ranged_Zero))
            car_dummy = np.concatenate((ranged_Zero, extra_data, ranged_Zero))
            walk_dummy = np.concatenate((ranged_Zero, ranged_Zero, extra_data))
            train_data = np.concatenate((train_data, pt_dummy), axis=1)
            train_data = np.concatenate((train_data, car_dummy), axis=1)
            train_data = np.concatenate((train_data, walk_dummy), axis=1)
            PT = np.expand_dims(CHOICE_PT, axis = 0)
            CAR = np.expand_dims(CHOICE_CAR, axis = 0)
            WALK = np.expand_dims(CHOICE_WALK, axis = 0)
            choices = np.concatenate((PT,CAR,WALK), axis = 0)
            choices = np.expand_dims(choices, axis = 1)
            train_data = np.concatenate((train_data, choices), axis=1)

    if fullArchitecture:
        behavior = df.iloc[:,42:95].values
#        behavior = df.iloc[:,42:48].values #Environment
        behavior = np.swapaxes(behavior,0,1)
        extra_data = np.concatenate((extra_data,behavior), axis=0)

        if dummyArchitecture:
            train_data = np.delete(train_data, -1, axis = 1)
            extra_range = extra_data.shape[0]
            ranged_Zero = np.expand_dims(np.array([ZEROs for i in range(extra_range)]), axis=0)
            extra_data = np.expand_dims(extra_data, axis=0)
            print(train_data.shape)
            print(ranged_Zero.shape)
            print(extra_data.shape)
            print(np.concatenate((extra_data, ranged_Zero, ranged_Zero)).shape)
            pt_dummy = np.concatenate((extra_data, ranged_Zero, ranged_Zero))
            car_dummy = np.concatenate((ranged_Zero, extra_data, ranged_Zero))
            walk_dummy = np.concatenate((ranged_Zero, ranged_Zero, extra_data))
            train_data = np.concatenate((train_data, pt_dummy), axis=1)
            train_data = np.concatenate((train_data, car_dummy), axis=1)
            train_data = np.concatenate((train_data, walk_dummy), axis=1)
            PT = np.expand_dims(CHOICE_PT, axis = 0)
            CAR = np.expand_dims(CHOICE_CAR, axis = 0)
            WALK = np.expand_dims(CHOICE_WALK, axis = 0)
            choices = np.concatenate((PT,CAR,WALK), axis = 0)
            choices = np.expand_dims(choices, axis = 1)
            print(choices.shape)
            train_data = np.concatenate((train_data, choices), axis=1)

    if simpleArchitecture:
        train_data = np.array(
                [[ZEROs,ZEROs, TimePT_scaled,  ZEROs, MarginalCostPT_scaled/ScaledIncome,  ZEROs, CHOICE_PT],
                [ASCs,  ZEROs, ZEROs, TimeCar_scaled, CostCarCHF_scaled/ScaledIncome,     ZEROs, CHOICE_CAR],
                [ZEROs, ASCs,  ZEROs, ZEROs,          ZEROs,                             distance_km_scaled, CHOICE_WALK]] )

        extra_data = np.array([age, HouseType,Gender, GenAbST,Education,FamilSitu,ScaledIncome,
                OwnHouse, Mothertongue, SocioProfCat, Student, Urban, NbChild, NbCar, Work, French, NbBicy ])

    if NNArchitecture:
        extra_data = np.array([TimePT_scaled, MarginalCostPT_scaled, ScaledIncome, CostCarCHF_scaled, TimeCar_scaled, distance_km_scaled,
        age, HouseType,Gender, GenAbST,Education,FamilSitu,ScaledIncome,
        OwnHouse, Mothertongue, SocioProfCat, Student, Urban, NbChild, NbCar, Work, French, NbBicy ])

    if inArchitecture:
        # train_data = np.array(
        #         [[ZEROs,ZEROs, TimePT_scaled,  ZEROs, MarginalCostPT_scaled/ScaledIncome,  ZEROs, CHOICE_PT],
        #         [ASCs,  ZEROs, ZEROs, TimeCar_scaled, CostCarCHF_scaled/ScaledIncome,     ZEROs, CHOICE_CAR],
        #         [ZEROs, ASCs,  ZEROs, ZEROs,          ZEROs,                             distance_km_scaled, CHOICE_WALK]] )
        #
        # # extra_data = np.array([HouseType_binaries[:,i] for i in range(HouseType_binaries[0].size)] +
        # # [Education_binaries[:,i] for i in range(Education_binaries[0].size)] +
        # # [FamilSitu_binaries[:,i] for i in range(FamilSitu_binaries[0].size)] +
        # # [SocioProfCat_binaries[:,i] for i in range(SocioProfCat_binaries[0].size)] +
        # # [DestAct_binaries[:,i] for i in range(DestAct_binaries[0].size)] +
        # # [ModeToSchool_binaries[:,i] for i in range(ModeToSchool_binaries[0].size)] +
        # # [TripPurpose_binaries[:,i] for i in range(TripPurpose_binaries[0].size)] +
        # #
        # # [age,Gender, GenAbST,
        # # ScaledIncome, OwnHouse, Mothertongue, Student, Urban, NbChild, NbCar,
        # # French, NbBicy, NbTransf,TypeCommune, FreqTripHouse, NbTrajects])

        extra_data = np.array([HouseType_binaries[:,i] for i in range(HouseType_binaries[0].size)] +
         [Education_binaries[:,i] for i in range(Education_binaries[0].size)] +
         [FamilSitu_binaries[:,i] for i in range(FamilSitu_binaries[0].size)] +
         [SocioProfCat_binaries[:,i] for i in range(SocioProfCat_binaries[0].size)] +
          [age,Gender, GenAbST ,ScaledIncome, OwnHouse, Mothertongue])



        # STRC Spread specification (bad)
        # extra_data = np.array([age, HouseType,Gender, GenAbST,Education,FamilSitu,
        # ScaledIncome, OwnHouse, Mothertongue, SocioProfCat, Student, Urban, NbChild, NbCar,
        # Work, French, NbBicy, NbTransf, DestAct,TypeCommune, FreqTripHouse, ModeToSchool, NbTrajects])


        if dummyArchitecture:
            train_data = np.delete(train_data, -1, axis = 1)
            extra_range = extra_data.shape[0]
            ranged_Zero = np.expand_dims(np.array([ZEROs for i in range(extra_range)]), axis=0)
            extra_data = np.expand_dims(extra_data, axis=0)

            pt_dummy = np.concatenate((extra_data, ranged_Zero, ranged_Zero))
            car_dummy = np.concatenate((ranged_Zero, extra_data, ranged_Zero))
            walk_dummy = np.concatenate((ranged_Zero, ranged_Zero, extra_data))
            train_data = np.concatenate((train_data, pt_dummy), axis=1)
            train_data = np.concatenate((train_data, car_dummy), axis=1)
            train_data = np.concatenate((train_data, walk_dummy), axis=1)
            PT = np.expand_dims(CHOICE_PT, axis = 0)
            CAR = np.expand_dims(CHOICE_CAR, axis = 0)
            WALK = np.expand_dims(CHOICE_WALK, axis = 0)
            choices = np.concatenate((PT,CAR,WALK), axis = 0)
            choices = np.expand_dims(choices, axis = 1)
            train_data = np.concatenate((train_data, choices), axis=1)



        print(extra_data.shape)
    train_data = np.swapaxes(train_data,0,2)
    extra_data = np.swapaxes(extra_data,0,1)

    if write:
        np.save(train_data_name, train_data)
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name, len(train_data[0])-1, extra_data[0].size



def biogeme_input(choices_num, filePath, fileInputName, saveExtension, extraInput, extensions = [''], biogemePath = 'biogeme/'):

    #extensions = ['']

    for extension in extensions:
        data = np.loadtxt(filePath+fileName+extension+'.dat',skiprows = 1)

        #Get input layer and Dense output values for Utility function
        model_keras = load_model(filePath + fileInputName+'_'+saveExtension+".h5")
        feature_input = model_keras.input

        new_feature = model_keras.get_layer(name = "Output new feature").output

        model = Model(feature_input, new_feature)

        #write headers:
        f = open(filePath + fileName+extension+'.dat','r')
        headers = f.readline()
        new_beta_header = ''
        for i in range(choices_num):
            new_beta_header += 'NEW_FEATURE_{}\t'.format(i+1)

        f.close()

        biogeme_data = open(filePath+biogemePath+fileInputName+'_'+saveExtension+extension+'.dat','wb')

        #biogeme_data.write(new_beta_header + headers)

        bins = []


        beta_num = 8
        choices_num = 3

        #exclusions:
        CHOICE = data[:,-7]
        #PURPOSE = data[:,4]
        exclude = (CHOICE == -1)
        exclude_list = [i for i,k in enumerate(exclude) if k > 0]

        data = np.delete(data,exclude_list, axis = 0)


        #Define:
        CHOICE = data[:,-7]
        PT_TT = data[:, 3]
        PT_COST = data[:,6]
        DISTANCE = data[:,-8]
        Gender = data[:,26]
        CAR_TT = data[:,8]
        CAR_CO = data[:,7]

        GENDER = [p if p == 1 else 0 for p in Gender]

        ASCs = np.ones(CHOICE.size)
        ZEROs = np.zeros(CHOICE.size)

        CHOICE_CAR = (CHOICE == 2)
        CHOICE_WALK = (CHOICE == 3)
        CHOICE_PT = (CHOICE == 1)


        train_data = np.array(
            [[ASCs, ZEROs, PT_TT, PT_COST, ZEROs, ZEROs, ZEROs, ZEROs, CHOICE_PT],
            [ZEROs,ASCs, ZEROs, ZEROs, CAR_TT, CAR_CO, GENDER, ZEROs, CHOICE_CAR],
            [ZEROs, ZEROs, ZEROs, ZEROs, ZEROs, ZEROs,  ZEROs, DISTANCE, CHOICE_CAR]] )

        train_data = np.swapaxes(train_data,0,2)
        train_data = np.expand_dims(train_data, -1)

        if extraInput:
            extra_data = np.delete(data,[18,19,21,22,25,26,27],axis = 1)
            extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

            # dirty fix for denseNN extra data (whose model only has 1 input)
            if len(model.input_shape) == 4:
                new_features = model.predict(extra_data)
            else:
                new_features = model.predict([train_data,extra_data])

        else:
            new_features = model.predict(train_data)

        new_features = np.squeeze(new_features)

        data = np.concatenate((new_features,data), axis = 1)

        np.savetxt(biogeme_data, data, fmt='%10.5f', header=new_beta_header+headers[:-1], delimiter = '\t', comments='')

        biogeme_data.close()
