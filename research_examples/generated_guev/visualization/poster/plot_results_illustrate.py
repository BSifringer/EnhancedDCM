import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import pickle

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


from utilities import grad_hess_utilities as ghu
from keras.models import load_model
from generated_guev import data_manager as dm


#import seaborn as sns

#sns.set()


#Dictionnary [neuron] of dictionaries: {betas, likelihood_train, likelihood_test}
encyclopedia = pickle.load(open('encyclo_mult.p', 'rb'))

neurons = [1, 2, 5, 10, 15, 25, 50, 100, 200, 500, 1001]
neurons = [ 2, 5, 10, 15, 25, 50, 100, 200, 500, 1001]
encyclopedia[1] = {}
encyclopedia[1]['betas'] = np.array([  1.6895992, 2.4960296])
encyclopedia[1]['likelihood_train'] =   3614.9331350326534
encyclopedia[1]['likelihood_test'] = 757.4825139045715


##Alternative Code
#def fetch_file(neuron):
#	return "../../Betas_generated_data_Enhancedscan{}extra.npy".format(neuron)
#	#return "Betas_Swissmetro_paper_EnhancednoDrop{}extra.npy".format(neuron)
#
#beta_list = []
#
#for neuron in neurons:
#	filename = fetch_file(neuron)
#	betas = np.load(filename)
#	n_betas = betas.size
#	beta_list.append(betas[:])
n_betas = 2

#Colors for Graph
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
#
# f, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(7,7))
# betas = []
# # first beta is ASC and is skipped
# for i in range(n_betas):
#
# 	beta_list = [encyclopedia[a]['betas'] for a in neurons]
# 	betas.append(np.abs(np.array([element[i] for element in beta_list])))
# 	ax1.semilogx(np.array(neurons), betas[i].flatten(), '.-', label = r"$\hat\beta_{}$".format(i+1))
#
# ax1.semilogx(np.array(neurons), np.ones(len(neurons))*2, '--', color = colors[0], label = r"$\beta_{1, true}$")
# ax1.semilogx(np.array(neurons), np.ones(len(neurons))*3, '--', color = colors[1], label = r"$\beta_{2, true}$")
# #ax1.semilogy(np.array(neurons), betas[1].flatten()/betas[0].flatten())
# ax1.set_xlabel('# of Neurons')
# ax1.set_ylabel('Beta Value')
# ax1.legend(loc = 7)
#
# likelihood_list = [encyclopedia[a]['likelihood_train'] for a in neurons]
# likelihood_test_list = [encyclopedia[a]['likelihood_test'] for a in neurons]
# print(likelihood_list)
# print(likelihood_test_list)
# ax2.semilogx(np.array(neurons), np.array(likelihood_test_list)/likelihood_test_list[0], '.-', label = "Test")
# ax2.semilogx(np.array(neurons), np.array(likelihood_list)/likelihood_list[0], '.-', label = "Train")
# ax2.set_ylabel("Normalized Likelihood")
# ax2.legend(loc = 0)
#
# plt.savefig('ToyScan3.eps', format = 'eps', dpi = 1000)
# plt.show()
#
# ---------------------------------------------
def fetch_model(neuron, path, extend):
	""" Load models from synthetic Neuron Scan """
	filename = "{}generated_0_{}{}.h5".format(path, neuron, extend)
	return load_model(filename)

def get_inputs_labels(filePath, fileInputName, filePart, simpleArchitecture=False, lmnlArchitecture=False, write=False):
	""" Get model inputs for each .dat for Train and Test """
	inputs_labels, extra, *_ = dm.keras_input(filePath, fileInputName, filePart, simpleArchitecture=simpleArchitecture,
											 lmnlArchitecture=lmnlArchitecture, write=write)

	labels = inputs_labels[:,-1,:]
	inputs = np.delete(inputs_labels, -1, axis = 1)
	inputs = np.expand_dims(inputs, -1)
	extra = np.expand_dims(extra,-1)
	extra = np.expand_dims(extra,-1)
	# extra = (extra - extra.mean(axis=0)) / extra.std(axis=0)
	return [inputs, extra], labels


case = "MNL_Full"
path, extend = ['../../poster/', 'MNL_Full']
path, extend = ['../../illustrate/', 'MNL_Full']

#model_inputs, train_labels = get_inputs_labels(path, 'generated_data', '_train')
#model_test_inputs, test_labels = get_inputs_labels(path, 'generated_data', '_test')
model_inputs, train_labels = get_inputs_labels(path, 'generated_0', '_train')
model_test_inputs, test_labels = get_inputs_labels(path, 'generated_0', '_test')

model = fetch_model("", path, extend)
betas = ghu.get_betas(model)

likelihood_train, accuracy_train = ghu.get_likelihood_accuracy(model, model_inputs[0], train_labels)
likelihood_test, accuracy_test = ghu.get_likelihood_accuracy(model, model_test_inputs[0], test_labels)
betas = ghu.get_betas(model)
stds = ghu.get_stds(model, model_inputs[0:1], train_labels)

print('\n\n----- Likelihood Train set -----')
likelihood = likelihood_train
print(case + ': {}'.format(np.array(likelihood)))

print('\n\n----- likelihood_test_ -----')
likelihood = likelihood_test
print(case + ': {}'.format(np.array(likelihood)))

print('\n\n----- Betas and Stds -----')
print('\n' + case + ': {}'.format(np.array(betas)))
print('stds' + ': {}'.format(np.array(stds)))
print('t-tests: {}'.format((np.array(betas) / np.array(stds))))

case = "L-MNL100"
path, extend = ['../../poster/', '100extra']
path, extend = ['../../illustrate/', '100extra']

#model_inputs, train_labels = get_inputs_labels(path, 'generated_data', '_train', lmnlArchitecture=True)
#model_test_inputs, test_labels = get_inputs_labels(path, 'generated_data', '_test', lmnlArchitecture=True)
model_inputs, train_labels = get_inputs_labels(path, 'generated_0', '_train', lmnlArchitecture=True)
model_test_inputs, test_labels = get_inputs_labels(path, 'generated_0', '_test', lmnlArchitecture=True)

model = fetch_model("Enhancedscan", path, extend)
betas = ghu.get_betas(model)

likelihood_train, accuracy_train = ghu.get_likelihood_accuracy(model, model_inputs, train_labels)
likelihood_test, accuracy_test = ghu.get_likelihood_accuracy(model, model_test_inputs, test_labels)
betas = ghu.get_betas(model)
stds = ghu.get_stds(model, model_inputs, train_labels)

print('\n\n----- Likelihood Train set -----')
likelihood = likelihood_train
print(case + ': {}'.format(np.array(likelihood)))

print('\n\n----- likelihood_test_ -----')
likelihood = likelihood_test
print(case + ': {}'.format(np.array(likelihood)))

print('\n\n----- Betas and Stds -----')
print('\n' + case + ': {}'.format(np.array(betas)))
print('stds' + ': {}'.format(np.array(stds)))
print('t-tests: {}'.format((np.array(betas) / np.array(stds))))
