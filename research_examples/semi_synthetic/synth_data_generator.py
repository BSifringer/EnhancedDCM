import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
def normalize(data):
    return (data-data.mean(axis=0))/data.std()

def generate_outcomes(filePart, illustrative, coef_list):
    a1, a2, a3, a4, a5, a6, a7 = coef_list
    fileName = 'swissmetro'
    data = np.loadtxt(fileName + filePart + '.dat', skiprows=1)
    beta_num = 4
    choices_num = 3

    # exclusions:
    CHOICE = data[:, -1]
    PURPOSE = data[:, 4]
    CAR_AV = data[:, 16]
    TRAIN_AV = data[:, 15]
    SM_AV = data[:, 17]

    exclude = ((CAR_AV == 0) + (CHOICE == 0) + (TRAIN_AV == 0) + (SM_AV == 0)) > 0
    exclude_list = [i for i,k in enumerate(exclude) if k > 0]

    data = np.delete(data, exclude_list, axis=0)

    # Define:
    CHOICE = data[:, -1]
    TRAIN_TT = data[:, 18] /100
    TRAIN_COST = data[:, 19] * (data[:, 12] == 0) /100 # if he owns a GA
    SM_TT = data[:, 21] /100
    SM_COST = data[:, 22] * (data[:, 12] == 0) /100 # if he owns a GA
    CAR_TT = data[:, 25]/100
    CAR_CO = data[:, 26]/100

    TRAIN_HE = data[:, 20]/100
    SM_HE = data[:, 23]/100
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

    x3 = normalize(DEST)
    x4 = normalize(AGE)
    x4 = AGE 
    x5 = normalize(ORIGIN)
    x6 = normalize(INCOME)
    x7 = normalize(PURPOSE)

    if illustrative:
        # ------------------- The Toy Function ------------------------------
        U_train = a1 * x1t + a2 * x2t    + a3 * x3 * x4 + a4 * x4 * x5
        U_SM =    a1 * x1s + a2 * x2s    + a3 * x3 * x4                 + a5 * x6 * x7
        U_car =   a1 * x1c + a2 * x2c                                                + a6 * x4 * x6 + a7 * x5 * x6
        # -------------------------------------------------------------------

    else:

        # ------------------- The Toy Function ------------------------------
        U_train = a1 * x1t + a2 * x2t    + a3 * x3**3 * x4 + a4 * x4**0.5 * x5
        U_SM =    a1 * x1s + a2 * x2s    + a3 * x3**3 * x4                 + a5 * x6 * x7**2
        U_car =   a1 * x1c + a2 * x2c                                                + a6 * x4 * x6**5 + a7 * x5**2 * x6**5
        # -------------------------------------------------------------------


    def multlogit_prob(U_a):
        return np.exp(U_a) / (np.array([np.exp(U_train), np.exp(U_SM), np.exp(U_car)]).sum(axis=0))

    P_train = multlogit_prob(U_train)
    P_SM = multlogit_prob(U_SM)
    P_car = multlogit_prob(U_car)
    plt.hist(np.concatenate([np.concatenate([P_train, P_SM]), P_car]))
    plt.show()
    #plt.hist(P_train)
    #plt.show()
    #plt.hist(P_car)
    #plt.show()
    #plt.hist(P_SM)
    #plt.show()
    return np.array([np.random.multinomial(1, [P_train[i], P_SM[i], P_car[i]]) for i in range(CHOICE.size)])


def saveFile(fileName, data, headers):
    file = open(fileName, 'wb')
    np.savetxt(file, data, fmt='%10.5f', header=headers, delimiter='\t', comments='')
    file.close()


def single_run(n, i, coeff, *args):
    """
    Creates and saves a dataset on generated outcomes
    input:
        - n: size of dataset
        - a: function coefficients
        - i: index of dataset
    """
    headers = "x1\tx2\tx3\tx4\tx5\tchoice"
    if unseen:
        headers = "x1\tx2\tx3\tx4\tx5\tx6\tx7\tchoice"

    outcomes, x1, x2, x3, x4, x5, *_ = generate_outcomes(n, coeff, *args)
    data = np.concatenate((x1, x2, x3, x4, x5, *_, outcomes), axis=1)
    #saveFile('generated_{}_train.dat'.format(i), data, headers)

    n_test = int(n * 0.2)
    outcomes_test, x1, x2, x3, x4, x5, *_ = generate_outcomes(n_test, coeff, *args)
    data_test = np.concatenate((x1, x2, x3, x4, x5, *_, outcomes_test), axis=1)
    #saveFile('generated_{}_test.dat'.format(i), data_test, headers)

    return data, data_test

    
if __name__ == "__main__":
    verbose = True

    def generate_set(filePath, illustrative, coef_list, verbose):
        synth_train_labels = generate_outcomes('_train', illustrative, coef_list)
        np.save(filePath+'synth_train_labels.npy', synth_train_labels)
        if verbose:
            print(np.array([label[0] == 1 for label in synth_train_labels]).sum())
            print(np.array([label[1] == 1 for label in synth_train_labels]).sum())
            print(np.array([label[2] == 1 for label in synth_train_labels]).sum())
        synth_test_labels = generate_outcomes('_test', illustrative, coef_list)
        np.save(filePath + 'synth_test_labels.npy', synth_test_labels)
    
    filePath = 'illustrative/'
    coef_list = [-1, -2, 0.2, 0.2, 0.5, -0.1, 0.1]
    generate_set(filePath, True, coef_list, verbose)

    filePath = 'power_log/'
    coef_list = [-1, -2, 1, -1, 3, 5, 2]
    generate_set(filePath, False, coef_list, verbose)
