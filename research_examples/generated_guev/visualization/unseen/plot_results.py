import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

sns.set()


encyclopedia = pickle.load(open('Encyclopedia_unseen.p', 'rb'))
n_range = 100

cases = [
#	'FULL_MNL_HYP',
	'MNL',
#	'FULL_MNL',
#	'MNL_HYP',
	'HYBRID',
#	'HRUSHCKA_MNL_HYP',
#	'HRUSHCKA_FULL_MNL_HYP'
	]

beta1_pos = {
	'FULL_MNL_HYP': 1,
	'MNL': 1,
	'FULL_MNL': 1,
	'TRUE_MNL': 0,
	'MNL_HYP': 1,
	'HYBRID': 0,
	'HRUSHCKA_MNL_HYP': 1,
	'HRUSHCKA_FULL_MNL_HYP': 1
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
	# b = abs(np.array(coefs_a2))>0.5
	# rel_error = rel_error[b]

	print(case + ': {}'.format(rel_error.mean()) + " +- {}".format(rel_error.std()))
	plt.plot(range(len(rel_error)), rel_error)
plt.show()

print('----- Rel Mean Error A1 and STD -----')
for case in cases:
	b1_pos = beta1_pos[case]
	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
	error = abs(np.array([beta[b1_pos] for beta in betas_final]) - np.array(coefs_a1))
	rel_error = abs(error/coefs_a1)
	#
	# a = abs(np.array(coefs_a1))>0.5
	# rel_error = rel_error[a]

	print(case + ': {}'.format(rel_error.mean()) + " +- {}".format(rel_error.std()))
	plt.plot(range(len(rel_error)), rel_error)
plt.show()


print('\n\n----- Mean Relative Error and STD A2/A1 -----')
for case in cases:
	b1_pos = beta1_pos[case]
	b2_pos = beta1_pos[case]+1
	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
#	error = abs(np.array([beta[2]/beta[1] for beta in betas_final]) - np.array(coefs_a2)/np.array(coefs_a1))
#	rel_error = abs(error/coefs_a2)
	error_a1 =  abs(np.array([beta[b1_pos] for beta in betas_final]) - np.array(coefs_a1))
	rel_error_a1 = abs(error_a1/coefs_a1)
	error_a2 = abs(np.array([beta[b2_pos] for beta in betas_final]) - np.array(coefs_a2))
	rel_error_a2 = abs(error_a2/coefs_a2)


	rel_error_ratio =  abs((rel_error_a1-rel_error_a2)/(1-rel_error_a1))

	#a = abs(np.array(coefs_a1))>0.5
	#b = abs(np.array(coefs_a2))>0.5

	#rel_error_ratio = rel_error_ratio[(b + a - ((b - a)))]
	#rel_error_ratio = rel_error_ratio[(rel_error_ratio<3)]
	#rel_print = [j if rel>3 else '' for j,rel in enumerate(rel_error_ratio) ]
	#print(rel_print)
	print(case + ': {}'.format(rel_error_ratio.mean()) + " +- {}".format(rel_error_ratio.std()))
	plt.plot(range(len(rel_error_ratio)), rel_error_ratio)
plt.legend(cases)
plt.show()

print('\n\n----- Likelihood Train set -----')
for case in cases:
	likelihood = [encyclopedia[a]['likelihood_train_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(likelihood).mean())+ " +- {}".format(np.array(likelihood).std()))

print('\n\n----- Likelihood Test set -----')
for case in cases:
	likelihood = [encyclopedia[a]['likelihood_test_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(likelihood).mean())+ " +- {}".format(np.array(likelihood).std()))

print('\n\n----- Accuracy Train set -----')
for case in cases:
	accuracy = [encyclopedia[a]['accuracy_train_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(accuracy).mean())+ " +- {}".format(np.array(accuracy).std()))

print('\n\n----- Accuracy Test set -----')
for case in cases:
	accuracy = [encyclopedia[a]['accuracy_test_' + case] for a in range(n_range)]
	print(case + ': {}'.format(np.array(accuracy).mean())+ " +- {}".format(np.array(accuracy).std()))



print('\n\n------- Percentage of t-test passed --------')
for case in cases:
	betas_final = [encyclopedia[a]['betas_'+case] for a in range(n_range)]
	stds = [encyclopedia[a]['stds_'+case] for a in range(n_range)]

	t_test1 = abs((np.array([beta[1] for beta in betas_final]) - np.array(coefs_a1))/np.array([std[1] for std in stds]))
	t_test2 = abs((np.array([beta[2] for beta in betas_final]) - np.array(coefs_a2))/np.array([std[2] for std in stds]))

	pass1 = [t<1.96 for t in t_test1]
	pass2 = [t<1.96 for t in t_test2]

	print(case + ': {}'.format((np.array(pass1).sum()+np.array(pass2).sum())/(2*len(pass1))))
