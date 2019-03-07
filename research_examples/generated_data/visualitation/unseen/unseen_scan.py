import sys, os
from multiprocessing import Process, Pool, cpu_count
from tqdm import tqdm
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
import Tensorflow.scan_utilities as su
if __name__ == '__main__':

	cases = [
		'MNL',
		'HYBRID',
]

	encyclopedia = {}
	n_range = 100
	fix_new = 1
	fix_all = 0

	paths_extends = {
		'MNL': ['../../unseen/','_MNL'],
		'FULL_MNL': ['../../unseen/','_MNL_Full'],
		'HYBRID': ['../../unseen/', '_Enhancedextra'],
		'coef':'../../unseen/'
		}  

	filePath = '../../unseen/'
	fileInputBase = 'generated'

	for number in tqdm(range(n_range)):
		encyclopedia[number] = su.scan_iteration(number, cases, paths_extends, filePath, fileInputBase, encyclopedia, std_flag = True )

	pickle.dump(encyclopedia, open('Encyclopedia_unseen.p', 'wb'))


