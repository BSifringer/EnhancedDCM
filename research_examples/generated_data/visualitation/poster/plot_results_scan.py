import matplotlib.pyplot as plt
import numpy as np
import pickle
#import seaborn as sns

#sns.set()

#Dictionnary [neuron] of dictionaries: {betas, likelihood_train, likelihood_test}
encyclopedia = pickle.load(open('encyclo_mult.p', 'rb'))

neurons = [1, 5, 10, 15, 25, 50, 100, 200, 500, 1001, 2000]


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
n_betas = 3

#Colors for Graph
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

f, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(7,7))

# first beta is ASC and is skipped
for i in range(n_betas-1):

	beta_list = [encyclopedia[a]['betas'] for a in neurons]
	betas_i = np.abs(np.array([element[i+1] for element in beta_list]))
	ax1.semilogx(np.array(neurons), betas_i.flatten(), '.-', label = r"$\hat\beta_{}$".format(i+1))

ax1.semilogx(np.array(neurons), np.ones(len(neurons))*2, '--', color = colors[0], label = r"$\beta_{1, true}$")
ax1.semilogx(np.array(neurons), np.ones(len(neurons))*3, '--', color = colors[1], label = r"$\beta_{2, true}$")
ax1.set_xlabel('# of Neurons')
ax1.set_ylabel('Beta Value')
ax1.legend(loc = 7)

likelihood_list = [encyclopedia[a]['likelihood_train'] for a in neurons]
likelihood_test_list = [encyclopedia[a]['likelihood_test'] for a in neurons]
print(likelihood_list)
print(likelihood_test_list)
ax2.semilogx(np.array(neurons), np.array(likelihood_test_list)/likelihood_test_list[0], '.-', label = "Test")
ax2.semilogx(np.array(neurons), np.array(likelihood_list)/likelihood_list[0], '.-', label = "Train")
ax2.set_ylabel("Normalized Likelihood")
ax2.legend(loc = 0)

plt.savefig('ToyScan3.eps', format = 'eps', dpi = 1000)
plt.show()
