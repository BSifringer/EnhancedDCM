import sys, os
from multiprocessing import Process, Pool, cpu_count

#Allows to run as main from any directory and import utility packages
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

import numpy as np
import pickle
import scan_utilities as su

if __name__ == '__main__':

	cases = [
#		'FULL_MNL_HYP',
		'MNL',
		'FULL_MNL',
#		'MNL_HYP',
		'HYBRID',
#		'HRUSHCKA_MNL_HYP',
#		'HRUSHCKA_FULL_MNL_HYP'
]

	encyclopedia = {}
	n_range = 120

	paths_extends = {
		'MNL': ['../../correlations/','_MNL'],
		'FULL_MNL': ['../../correlations/', '_MNL_Full'],
		'HYBRID': ['../../correlations/', '_Enhancedextra'],
		'coef': '../../correlations/'
		}  

	filePath = '../../correlations/'
	fileInputBase = 'generated'

	for number in range(n_range):
		encyclopedia[number] = su.scan_iteration(number, cases, paths_extends, filePath, fileInputBase, encyclopedia, std_flag = False )

	pickle.dump(encyclopedia, open('Encyclopedia_correlations.p', 'wb'))
