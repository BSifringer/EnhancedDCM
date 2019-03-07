import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns # For Figure Style

"""
	Plot Swissmetro RealDataset L-MNL neuron Scan. Loads previously saved dictionnary
"""


""" Figure Settings """
sns.set()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 40}
#
#matplotlib.rc('font', **font)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
f, (ax2, ax1, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7,7))

""" Prepare Data (Add MNL) """
encyclopedia = pickle.load(open('encyclo_mult.p', 'rb'))
encyclopedia[1] = {}

#LOG-SCALE, put Neuron 0 in place of Neuron 1 ( = MNL Results: )
encyclopedia[1]['betas'] = np.array([ 0,0,-1.34, -0.695, -0.733 , 0, 0, 0, 0])
encyclopedia[1]['likelihood_train'] = 5764
encyclopedia[1]['likelihood_test'] = 1433

labels = ['time','cost','freq']
beta_list = []
betas = []
n_betas = encyclopedia[5]['betas'].size

""" Plot """
neurons = [1, 5, 10, 15, 25, 50, 100, 200, 500, 1001, 2000, 5000]
for i in range(n_betas-2):

	beta_list = [encyclopedia[a]['betas'] for a in neurons]
	#print(beta_list)
	betas.append(np.array([element[i+2] for element in beta_list]))
	#print(betas[i])
	label = "eta_{" + labels[i] + "}"
	ax1.semilogx(np.array(neurons), betas[i].flatten(), '.-', label = r"$\b{}$".format(label))

ax3.semilogx(np.array(neurons), betas[1].flatten()/betas[0].flatten(), '.-', label = "VOT")
ax3.semilogx(np.array(neurons), betas[1].flatten()/betas[2].flatten(), '.-', label = "VOF")
ax3.legend(loc = 2)
ax3.set_ylabel('Ratio Values')



ax3.set_xlabel('# of Neurons')
ax1.set_ylabel('Beta Values')
ax1.legend(loc = 3)

likelihood_list = [encyclopedia[a]['likelihood_train'] for a in neurons]
likelihood_test_list = [encyclopedia[a]['likelihood_test'] for a in neurons]
print(likelihood_list)
print(likelihood_test_list)
ax2.semilogx(np.array(neurons), np.array(likelihood_test_list)/likelihood_test_list[0], '.-', label = "Test")
ax2.semilogx(np.array(neurons), np.array(likelihood_list)/likelihood_list[0], '.-', label = "Train")
ax2.set_ylabel("Normalized Likelihood")
ax2.legend(loc = 3 )
plt.savefig('SwissScan3.eps', format = 'eps', dpi = 1000)

plt.show()