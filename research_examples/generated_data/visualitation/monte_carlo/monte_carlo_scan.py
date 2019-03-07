from tqdm import tqdm


if __name__ == "__main__" and __package__ is None:
    """Allows to run as main from any directory and import utility packages"""
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

import numpy as np
import pickle
import scan_utilities as su

"""
    Creates + saves dictionary and for monte carlo experiments 
    How To:
    -------
    	- Comment out cases from case list if needed
    	- Change std flag to remove std estimation, and speed up computation
"""

std_computation = True
n_range = 100 	# Number of Monte Carlo experiments

if __name__ == '__main__':

	cases = [
		'FULL_MNL_HYP',
		'MNL',
		'FULL_MNL',
		'TRUE_MNL',
		'MNL_HYP',
		'HYBRID',
		'HRUSHCKA_MNL_HYP',
		'HRUSHCKA_FULL_MNL_HYP',
		'HRUSCHKA2004',
		]

	paths_extends = {
		'FULL_MNL_HYP': ['../../monte_carlo/','_Enhanced_Full'],
		'MNL': ['../../monte_carlo/','_MNL'],
		'FULL_MNL': ['../../monte_carlo/','_MNL_Full'],
		'TRUE_MNL': ['../../monte_carlo/','_MNL_True'],
		'MNL_HYP': ['../../monte_carlo/','_Enhanced'],
		'HYBRID': ['../../monte_carlo/', '_Enhanced_extra'],
		'HRUSHCKA_MNL_HYP': ['../../monte_carlo_Hruschka/', '_Enhanced'],
		'HRUSHCKA_FULL_MNL_HYP': ['../../monte_carlo_Hruschka/', '_Enhanced_Full'],
		'HRUSCHKA2004': ['../../monte_carlo_Hruschka/', '_NN_Big_Full'],
		'coef':'../../monte_carlo/'
		}

	encyclopedia = {}
	filePath = '../../monte_carlo/'
	fileInputBase = 'generated'

	for number in tqdm(range(n_range)):
		encyclopedia[number] = su.scan_iteration(number, cases, paths_extends, filePath, fileInputBase, encyclopedia, std_flag = std_computation )

	pickle.dump(encyclopedia, open('Encyclopedia_monte_carlo.p', 'wb'))