import matplotlib.pyplot as plt
import numpy as np
import pickle
#import seaborn as sns

#sns.set()
import numpy as np
import pickle

if __name__ == "__main__" and __package__ is None:
	""" Imports fix, Call function in it's directory """
	from sys import path
	from os.path import dirname as dir
	path.append(dir(path[0]))
	splits = path[0].split('/')

	parent = '/'.join(splits[:-4])
	path.append(dir(parent))
	parent = '/'.join(splits[:-3])
	path.append(dir(parent))
	parent = '/'.join(splits[:-2])
	path.append(dir(parent))
	parent = '/'.join(splits[:-1])
	path.append(dir(parent))

	__package__ = "generated_data"

from EnhancedDCM.utilities import grad_hess_utilities as ghu
from medical import data_manager as dm
from keras.models import load_model
from keras import backend as K
fileName = 'optima'
cases = {
		"MNL_true" : ['../spread/', '_MNL_true_', ""],
		"L-MNL_true" : ['../spread/', '_Enhanced_correl_', "extra"],
		"MNL_endo" : ['../spread/', '_MNL_endo_', ""],
		"L-MNL" : ['../spread/', '_Enhanced', "extra"],
		"MNL_latent" : ['../spread/', '_MNL_latent_', ""],
		}

cases = ['MNL', 'L-MNL100', 'L-MNL100_simple', 'L-MNL30', 'L-MNL30_simple', 'NN100', 'NN30', 'L-MNL_in']
ticks_names = cases
cases = ['MNL', 'L-MNL30', 'L-MNL30_simple', 'NN30']
cases = ['MNL', 'L-MNL100', 'L-MNL100_simple', 'NN100', 'NN30']
ticks_names = ['MNL', 'L-MNL', 'L-MNL'+r'$_2$', 'NN', 'NN'+'$_{40}$']
n_range = 100
load = False

data = {}

dict = pickle.load(open('../spread/spread.p','rb'))

# data = {key:dict[key]['loss_train'] for key in cases}
data = {key:dict[key]['acc_train'] for key in cases}
p_data = np.array([np.array(array) for array in data.values()])
p_data = np.swapaxes(p_data, 0,1)

# data_2 = {key:dict[key]['loss_test'] for key in cases}
data_2 = {key:dict[key]['acc_test'] for key in cases}
p_data_2 = np.array([np.array(array) for array in data_2.values()])
p_data_2 = np.swapaxes(p_data_2, 0,1)
ticks = list(ticks_names)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(p_data, positions=np.array(range(len(p_data[0])))*2.0-0.4, sym='', widths=0.6, showmeans=True)
bpr = plt.boxplot(p_data_2, positions=np.array(range(len(p_data_2[0])))*2.0+0.4, sym='', widths=0.6, showmeans=True)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Train loss')
plt.plot([], c='#2C7BB6', label='Test loss')
plt.hlines(0.5233, -1., len(ticks)*2-1, linestyles='dashed', color='black')
# plt.hlines(0.6492, -1., len(ticks)*2-1, linestyles='dashed', color='black')
plt.legend(loc=6)

plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=30)
plt.xlim(-1.5, len(ticks)*2-.5)
#plt.ylim(0.35, 0.65)
# plt.tight_layout()
#plt.savefig('boxcompare.png')
plt.ylabel('Average Loss')

plt.savefig('box_paper_acc.eps', format = 'eps', dpi = 1000)

plt.show()
