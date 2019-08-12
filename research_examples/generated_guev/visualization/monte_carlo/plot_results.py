import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
#import seaborn as sns
#sns.set() 				# Changes Figures style

"""
    Prints and plots model results and statistics
    How To:
    -------
        - Choose your dictionary to load
        - Choose cases that appear in dictionnary
"""


current_dir = os.path.abspath(os.curdir)
encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo.p', 'rb'))
encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo_Hrusch.p', 'rb'))
# encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo_LMNL.p', 'rb'))
#encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo_cluster.p', 'rb'))
# encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo_NN2004.p', 'rb'))
#encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo_true.p', 'rb'))
#encyclopedia = pickle.load(open(current_dir+'/Encyclopedia_monte_carlo_500.p', 'rb'))

n_range = len(encyclopedia)

cases = [
	# 'FULL_MNL_HYP',
	# 'MNL',
	# 'FULL_MNL',
	# 'TRUE_MNL',
	# 'MNL_HYP',
	# 'HYBRID',
	# 'HRUSHCKA_MNL_HYP',
	'HRUSHCKA_FULL_MNL_HYP',
	'HRUSHCKA_FULL_MNL_HYP_100'
	]
# cases = [
# 	'FULL_MNL_HYP',
# 	'MNL',
# 	'FULL_MNL',
# 	'TRUE_MNL',
# 	'MNL_HYP',
# 	'HYBRID',
# 	'HRUSHCKA_MNL_HYP',
# 	'HRUSHCKA_FULL_MNL_HYP',
# 	'HRUSHCKA_FULL_MNL_HYP_100'
# 	]
# NN cases have no betas, so only used for Likelihood
casesNN = cases
NN_case = 'HRUSCHKA2004'
# casesNN = cases + [NN_case]

# Position of first Beta, based on architecture
beta1_pos = {
	'FULL_MNL_HYP': 1,
	'MNL': 1,
	'FULL_MNL': 1,
	'TRUE_MNL': 0,
	'MNL_HYP': 1,
	'HYBRID': 0,
	'HRUSHCKA_MNL_HYP': 1,
	'HRUSHCKA_FULL_MNL_HYP': 1,
	'HRUSHCKA_FULL_MNL_HYP_100': 1
}

coefs = [encyclopedia[a]['coef'] for a in range(n_range)]
coefs_a1 = [coef[0] for coef in coefs]
coefs_a2 = [coef[1] for coef in coefs]

error_dict = {}

print('----- Rel Mean Error A2 and STD -----')
for case in cases:
	b2_pos = beta1_pos[case]+1
	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
	error = abs(np.array([beta[b2_pos] for beta in betas_final]) - np.array(coefs_a2))
	rel_error = abs(error/coefs_a2)

	print(case + ': {}'.format(rel_error.mean()) + " +- {}".format(rel_error.std()))
	plt.plot(range(len(rel_error)), rel_error)
plt.show()

print('\n\n----- Rel Mean Error A1 and STD -----')
for case in cases:
	b1_pos = beta1_pos[case]
	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
	error = abs(np.array([beta[b1_pos] for beta in betas_final]) - np.array(coefs_a1))
	rel_error = abs(error/coefs_a1)

	print(case + ': {}'.format(rel_error.mean()) + " +- {}".format(rel_error.std()))
	plt.plot(range(len(rel_error)), rel_error)
plt.show()


print('\n\n----- Mean Relative Error and STD A2/A1 -----')
for case in cases:
	b1_pos = beta1_pos[case]
	b2_pos = beta1_pos[case]+1
	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
	error_a1 =  abs(np.array([beta[b1_pos] for beta in betas_final]) - np.array(coefs_a1))
	rel_error_a1 = abs(error_a1/coefs_a1)
	error_a2 = abs(np.array([beta[b2_pos] for beta in betas_final]) - np.array(coefs_a2))
	rel_error_a2 = abs(error_a2/coefs_a2)
	rel_error_ratio =  abs((rel_error_a1-rel_error_a2)/(1-rel_error_a1))

	print(case + ': {}'.format(rel_error_ratio.mean()) + " +- {}".format(rel_error_ratio.std()))
	plt.plot(range(len(rel_error_ratio)), rel_error_ratio)
plt.legend(cases)
plt.show()


print('\n\n----- Accuracy Train set -----')
for case in casesNN:
	accuracy = [encyclopedia[a]['accuracy_train_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(accuracy).mean())+ " +- {}".format(np.array(accuracy).std()))

print('\n\n----- Accuracy Test set -----')
for case in casesNN:
	accuracy = [encyclopedia[a]['accuracy_test_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(accuracy).mean())+ " +- {}".format(np.array(accuracy).std()))




print('\n\n----- Likelihood Train set -----')
for case in casesNN:
	likelihood = [encyclopedia[a]['likelihood_train_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(likelihood).mean())+ " +- {}".format(np.array(likelihood).std()))

print('\n\n----- likelihood_test_ -----')
for case in casesNN:
	likelihood = [encyclopedia[a]['likelihood_test_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(likelihood).mean())+ " +- {}".format(np.array(likelihood).std()))

print('\n\n------- Percentage of t-test passed --------')
for case in cases:
	b1_pos = beta1_pos[case]
	b2_pos = beta1_pos[case]+1

	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
	stds = [encyclopedia[a]['stds_'+case] for a in range(n_range)]

	beta1 = np.array([beta[b1_pos] for beta in betas_final])
	beta2 = np.array([beta[b2_pos] for beta in betas_final])
	std1 = np.array([std[b1_pos] for std in stds])
	std2 = np.array([std[b2_pos] for std in stds])

	t_test1 = abs((beta1 - np.array(coefs_a1))/std1)
	t_test2 = abs((beta2 - np.array(coefs_a2))/std2)

	pass1 = [t<1.96 for t in t_test1]
	pass2 = [t<1.96 for t in t_test2]

	print(case + ': {}'.format((np.array(pass1).sum()+np.array(pass2).sum())/(2*len(pass1))))

	# b_ratio = beta2/beta1
	# std_ratio = (1/(beta1**2)*(std2**2) + (beta2**2)/(beta1**4)*(std1**2))**0.5
	# t_test_ratio = abs((b_ratio - np.array(coefs_a2)/np.array(coefs_a1))/std_ratio)
	# pass_ratio = [t<1.96 for t in t_test_ratio]
	# print(case +'ratio : {}'.format(np.array(pass_ratio).sum()/len(pass_ratio)))
	b_ratio = beta1/beta2
	std_ratio = (1/(beta2**2)*(std1**2) + (beta1**2)/(beta2**4)*(std2**2))**0.5
	t_test_ratio = abs((b_ratio - np.array(coefs_a1)/np.array(coefs_a2))/std_ratio)
	pass_ratio = [t<1.96 for t in t_test_ratio]
	print(case +'ratio : {}'.format(np.array(pass_ratio).sum()/len(pass_ratio)))
