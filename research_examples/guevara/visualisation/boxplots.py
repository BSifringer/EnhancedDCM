import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

sns.set()
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
fileName = 'guevara'
cases = {
		"MNL_true" : ['../scan/', '_MNL_true_', ""],
		# "L-MNL_true" : ['../scan/', '_Enhanced_correl_', "extra"],
		"L-MNL_true100" : ['../scan/', '_Enhanced_correl100_', "extra"],
		"MNL_endo" : ['../scan/', '_MNL_endo_', ""],
		# "L-MNL" : ['../scan/', '_Enhanced', "extra"],
		# "L-MNL100" : ['../scan/', '_Enhanced100_', "extra"],
		# "MNL_latent" : ['../scan/', '_MNL_latent_', ""],
		# "wL-MNL" : ['../scan/', '_Enhanced_weak_', "extra"],
		# "2wL-MNL" : ['../scan/', '_Enhanced2lay_weak_', "extra"],
		# "mL-MNL" : ['../scan/', '_Enhanced_multi_', "extra"],
		# "2mL-MNL" : ['../scan/', '_Enhanced2lay_multi_', "extra"]
	#	"MNL_nolog" : ['../scan_log/', '_MNL_true_', ""],
	#	"L-MNL_truelog" : ['../scan_log/', '_Enhanced_correl_', "extra"],
	#	"MNL_nolog" : ['../scan_exp/', '_MNL_true_', ""],
	#	"L-MNL_truelog" : ['../scan_exp/', '_Enhanced_correl_', "extra"],
	#	"MNL_endo" : ['../scan/', '_MNL_endo_', ""],
	#	"MNL_latent" : ['../scan/', '_MNL_latent_', ""],
	#	"L-MNL" : ['../scan_log/', '_Enhanced', "extra"],
			# "MNL_true" : ['../scan_interact/', '_MNL_true_', ""],
			# "L-MNL_true" : ['../scan_interact/', '_Enhanced_correl_', "extra"],
			# "MNL_endo" : ['../scan_interact/', '_MNL_endo_', ""],
			# "L-MNL" : ['../scan_interact/', '_Enhanced', "extra"],
			# "MNL_latent" : ['../scan_interact/', '_MNL_latent_', ""]
		}

names = {
		"MNL_true" : r'$MNL_{true}$',
		"L-MNL_true" : r'$L$-$MNL_{true}$',
		"L-MNL_true100" : r'$L$-$MNL_{true,100}$',
		"MNL_endo" : r'$MNL_{endo}$',
		"L-MNL" : r'$L$-$MNL_{ind}$',
		"L-MNL100" : r'$L$-$MNL_{ind,100}$',
		"MNL_latent": r'$MNL_{ind}$',
		"wL-MNL" : "wL-MNL",
		"2wL-MNL" :"2wL-MNL" ,
		"mL-MNL" : "mL-MNL",
		"2mL-MNL" :"2mL-MNL"
	}
n_range = 100
load = False

data = {}


if not load:
	for (case, [path, model_name, extension]) in cases.items():
		data[case] = []
		for i in range(n_range):
			model = load_model(path+fileName+model_name+"{}".format(i)+extension + ".h5")
			betas = ghu.get_betas(model)
			data[case].append(betas[1]/betas[2])
			K.clear_session()

	pickle.dump(data, open('ratios3.p', 'wb'))
else:
	data = pickle.load(open('ratios3.p', 'rb'))

p_data = np.array([array for i, array in enumerate(data.values()) if i<len(cases)])
print(p_data.shape)
p_data = np.swapaxes(p_data, 0,1)
# plt.boxplot(p_data, whis = 100, showmeans = True, labels = [names[case] for case in cases.keys()])
plt.boxplot(p_data, showmeans = True, labels = [names[case] for case in cases.keys()])
plt.hlines(-2, 0.1, 4.9, linestyles='dashed')
plt.ylim(-3,-0.5)
plt.ylabel(r'$\hat{\beta}_p/\hat{\beta}_a$', position=(0,1.02), rotation=0)
plt.savefig('guevara2.eps', format = 'eps', dpi = 1000)
plt.show()
