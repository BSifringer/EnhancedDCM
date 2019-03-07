import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import sys
sns.set()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

encyclopedia = pickle.load(open('Encyclopedia_correlations_cluster.p', 'rb'))
n_range = 900

cases = [
#	'FULL_MNL_HYP',
	'MNL',
	'FULL_MNL',
#	'MNL_HYP',
	'HYBRID',
#	'HRUSHCKA_MNL_HYP',
#	'HRUSHCKA_FULL_MNL_HYP'
]

coefs = [encyclopedia[a]['coef'] for a in range(n_range)]
coefs_a1 = [coef[0] for coef in coefs]
coefs_a2 = [-coef[1] for coef in coefs]

legend ={
	'MNL':'Logit'+r'$(\mathcal{X}_2)$',
	'FULL_MNL':'Logit'+r'$(\mathcal{X}_1)$',
	'HYBRID':'L-MNL'+r'$(100,\mathcal{X},\mathcal{Q})$'
}
error_dict = {}

w = 0.2
delta = -w
ax = plt.subplot(111)
it = 100
p_list = [1, 0.95, 0.9, 0.85,  0.8, 0.6, 0.4, 0.2, 0.0]
for c, case in enumerate(cases):
	for i,p in enumerate(p_list):
		x_low = it*i
		x_high = it*(i+1)
		betas_final = [encyclopedia[a]['betas_'+case] for a in range(x_low,x_high)]
		a1 = coefs_a1[x_low:x_high]
		a2 = coefs_a2[x_low:x_high]
		error_a1 = abs(np.array([beta[1] for beta in betas_final]) - np.array(a1))
		rel_error_a1 = abs(error_a1/(np.array(a1)))
		error_a2 = abs(np.array([beta[2] for beta in betas_final]) - np.array(a2))
		rel_error_a2 = abs(error_a2/(np.array(a2)))

		rel_error = abs((rel_error_a1-rel_error_a2)/(1-rel_error_a2))

		print(case + ': {}'.format(rel_error.mean()) + " ± {}".format(rel_error.std()))
		if i == 0:
			ax.bar(i+delta,100*rel_error.mean(), width=w, color=colors[c], label = legend[case])
		else:
			ax.bar(i+delta,100*rel_error.mean(), width=w, color=colors[c])

	delta = delta+w
plt.xticks(np.arange(len(p_list)),p_list)
plt.xlabel('Correlation coefficient p')
plt.ylabel(r'$\beta_1/\beta_2$'+' Relative Error [%]')
plt.legend()
plt.savefig('Correlation1_2.png', format='png', dpi=600)
plt.show()


delta = -w
ax = plt.subplot(111)
p_list = [1, 0.95, 0.9, 0.85,  0.8, 0.6, 0.4, 0.2, 0.0]
for c, case in enumerate(cases):
	for i,p in enumerate(p_list):
		x_low = it*i
		x_high = it*(i+1)
		betas_final = [encyclopedia[a]['betas_'+case] for a in range(x_low,x_high)]
		a1 = coefs_a1[x_low:x_high]
		a2 = coefs_a2[x_low:x_high]
		error = abs(np.array([beta[1] for beta in betas_final]) - np.array(a1))
		rel_error = abs(error/(np.array(a1)))

		print(case + ': {}'.format(rel_error.mean()) + " ± {}".format(rel_error.std()))
		if i == 0:
			ax.bar(i+delta,100*rel_error.mean(), width=w, color=colors[c], label = legend[case])
		else:
			ax.bar(i+delta,100*rel_error.mean(), width=w, color=colors[c])

	delta = delta+w
plt.xticks(np.arange(len(p_list)),p_list)
plt.xlabel('Correlation coefficient p')
plt.ylabel(r'$\beta_1$'+' Relative Error [%]')
plt.legend()
plt.savefig('Correlation1.png', format='png', dpi=600)
plt.show()


delta = -w
ax = plt.subplot(111)
p_list = [1, 0.95, 0.9, 0.85,  0.8, 0.6, 0.4, 0.2, 0.0]
for c, case in enumerate(cases):
	for i,p in enumerate(p_list):
		x_low = it*i
		x_high = it*(i+1)
		betas_final = [encyclopedia[a]['betas_'+case] for a in range(x_low,x_high)]
		a1 = coefs_a1[x_low:x_high]
		a2 = coefs_a2[x_low:x_high]
		error = abs(np.array([beta[2] for beta in betas_final]) - np.array(a2))
		rel_error = abs(error/(np.array(a2)))

		print(case + ': {}'.format(rel_error.mean()) + " ± {}".format(rel_error.std()))
		ax.bar(i+delta,rel_error.mean(), width=w, color=colors[c])
		if i == 0:
			ax.bar(i+delta,100*rel_error.mean(), width=w, color=colors[c], label = legend[case])
		else:
			ax.bar(i+delta,100*rel_error.mean(), width=w, color=colors[c])
	delta = delta+w
plt.xticks(np.arange(len(p_list)),p_list)
plt.xlabel('Correlation coefficient p')
plt.ylabel(r'$\beta_2$'+ 'Relative Error [%]')
plt.legend()
plt.show()