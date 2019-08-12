import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-3])
    path.append(dir(parent))
    parent = '/'.join(splits[:-2])
    path.append(dir(parent))
    parent = '/'.join(splits[:-1])
    path.append(dir(parent))

    __package__ = "generated_data"

from EnhancedDCM.utilities import grad_hess_utilities as ghu
from swissmetro_paper import data_manager as swissDM


""" L-MNL on SwissMetro study ; Saliency Map and Sensitivity Analyisis """

sns.set()
lmnlArchitecture = True
beta_num = 3
nExtraFeatures = 15


""" Load Model and Data"""
_, _, train_data_name = swissDM.keras_input('../models/', 'swissmetro', filePart='_test',
                                            lmnlArchitecture=lmnlArchitecture)

model_name = '../models/swissmetro_Enhancedextra.h5'
train_name = '../models/keras_input_swissmetro_noASC_train.npy'
extra_name = '../models/keras_input_swissmetro_noASC_train_extra.npy'
model = load_model(model_name)

train_data = np.load(train_name)
labels = train_data[:,-1,:]
train_data = np.delete(train_data, -1, axis = 1)
train_data = np.expand_dims(train_data, -1)

extra_data = np.load(extra_name)
extra_data = np.expand_dims(extra_data,-1)
extra_data = np.expand_dims(extra_data,-1)

extra_data = (extra_data-extra_data.mean(axis=0))/extra_data.std(axis=0)

#model_inputs = [train_data]
model_inputs = [train_data, extra_data]


""" Plot Saliency Map """
# layer_name = 'Utilities'
# inputs_indice = 1
#
# invHess = ghu.get_inverse_Hessian(model, model_inputs, labels, layer_name)
# print("The Hessian is Symmetric: \n {} \n".format(np.linalg.inv(invHess)))
# print("The inverse diagonal has variance terms: \n {} \n".format(invHess))
# indices = np.array([i for i,label in enumerate(labels) if label[1]==1])
# print(model.evaluate([train_data[indices], extra_data[indices]], labels[indices]))
# labels = model.predict([train_data, extra_data])
# for label in labels:
#     for i in range(label.size):
#         label[i] = label[i]==np.max(label)
#
# old = np.zeros(12)
# for i in range(train_data.shape[2]):
#
#     heatmap = ghu.get_inputs_gradient(model, model_inputs, [label if label[i]==1 else np.array([0,0,0]) for label in labels], inputs_indice)
#
#     count = 0
#     for label in labels:
#         if label[i]==1:
#             count = count+1
#     grad_sum = (np.sum(abs(heatmap), axis=0))/count
#     print("Alternative {}: {} predicted count".format(i, count))
#     print("Sum of all abs. gradients per input/target :\n {} \n".format(grad_sum))
#     plt.bar([i for i in range(grad_sum.size)], grad_sum, bottom=old)
#     old = old+grad_sum
#     print("Mean of gradients per input/target:\n {} \n".format(np.mean((heatmap), axis=0)))
# plt.xticks([i for i in range(grad_sum.size)],['PURPOSE', 'FIRST', 'TICKET', 'WHO', 'LUGGAGE', 'AGE', 'MALE', 'INCOME', 'GA', 'ORIGIN', 'DEST','SM_SEATS'])
# plt.xticks(fontsize=9, rotation=60)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.legend(['Train', 'SM', 'Car'])
# plt.ylabel('Feature impact on utility functions')
# plt.tight_layout()
# plt.savefig('feature_impact_score_abs', format='png', dpi=1200)
# plt.show()


""" Plot Sensitivity Analysis """

n_inputs = 10
n_samples = 100
names = ['PURPOSE', 'FIRST', 'TICKET', 'WHO', 'LUGGAGE', 'AGE', 'MALE', 'INCOME', 'GA', 'ORIGIN', 'DEST','SM_SEATS']
for i in [5,7]:
    x, predictions = ghu.elasticity_study(model, [train_data.copy(), extra_data.copy()], inputs_indice=1, feature_indice=i, n=n_samples, x_range=[-0.5,0.5])
    plt.plot(x*100,np.array(predictions)/extra_data.shape[0]*100)
    plt.xlabel('% Change in Feature {}'.format(names[i]))
    plt.ylabel('% Change in Mode Shares')
    plt.legend(['Train', 'SM', 'Car'])
    plt.tight_layout()
    plt.savefig('count_feature_{}.png'.format(i), format='png', dpi=1000)
    plt.show()
