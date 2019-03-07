# EnhancedDCM
Enhancing Discrete Choice Models with Learning Representation : The Learning MultiNomial Logit

This is the original keras implementation of L-MNL with examples used in our paper: [Let Me Not Lie: Learning MultiNomial Logit](https://arxiv.org/abs/1812.09747). 


## Prerequisites

The code successfully runs with:
* Python 3.6.3
* Tensorflow 1.5.0
* Keras 2.2.4

### Libraries

For visualization, you will also need these python packages:
* matplotlib
* seaborn

## Running a dataset
In utilities, there are the common scripts for any datasets. 
* Models are individually defined in `models.py`
* Data loading and training is done with `train_utils.py`
* `run_utils.py` is a helper which compiles the models and sends them to training. 
* `grad_hess_utilities.py` is used for investigating and visualizing the trained models. 

Every dataset has its own folder and main run script.

In the folder you will find:
* `data_manager.py` - This is the most important script for each experiment. It must read through your dataset and prepare the inputs for the model. This is where the utility functions are defined with the input set X and where we prepare the Neural network features Q.
    * The first input for the utilities must be of dimension: [#individuals x (beta number + 1) x choice number]. The added +1 in the second dimension is the label, 1 or 0, wether the alternative was chosen or not. 
    * The second input for the neural network component must be of dimension: [#individuals x Q_features x 1]
    * Caution: The code was made modular by giving flexibility to paths and file names. Exception lies with the naming convention of the inputs. The second input must have the same name as the first, say 'xx.npy', but with an added '_extra'. As such, we get: 'xx_extra.npy' for the name of the second input.
* the dataset or scripts to generate the dataset
* folders to contain the various experiments (datasets, trained models, ...)
* data and model visualization scripts

In the run script you will find:
* flags to set on or off depending on the experiment you wish to run
* the code to create, train and save desired models

For example, the swissmetro experiment has all necessary datasets on the Master branch and you may simply call:

```
cd ready_example/
python3 swissmetro_paper_run --models --scan
```

All other experiments need to have their necessary datasets generated first as explained below. This is to avoid excessive memory size of this git folder.


## Generate the synthetic datasets

For the fully synthetic data experiments, there is a shell script which generates all the necessary data (~100Mb). Simply run from root: 
```
cd research_examples/generated_data/
.generate_all.sh
```

For the semi-synthetic data experiment, run:

```
cd research_examples/semi_synthetic/
python3 synth_data_generator.py 
```

you can now run the experiments in the folder `research_examples/` with: 

```
python3 generated_run.py --scan --mc --mc_hr --corr --unseen
```
or
```
python3 semi_synthetic_run.py
```


## Add your own dataset

Goals:
* Make your own adapted `data_manager.py`
* Make your own main run script
* Optional: Tweak scripts in utilities to change optimizers, add your models, add cross validation methods, etc.. 


The key for training your own dataset on L-MNL is splitting features into 2 sets, X and Q, and then use the common utilities with a main script. To do this, X is of shape [# individuals x (beta number + 1) x choice number] and corresponds to the utility functions. The added +1 in the first dimension is the label, 1 or 0, wether the alternative was chosen or not. Q is of shape [# individuals x Q_features x 1].

In the given examples, this is done in their respective `data_manager.py`, keras_input() function, where the two sets are saved as vectors in 'xx.npy' and 'xx_extra.npy', and the name 'xx' is returned. When the name is given to a run_utils.py function, it will train the corresponding model to your dataset. This is done in the main script for each dataset with names ending with '_run.py', where we call upon keras_input() with an architecture and data input specific to each experiment, and then train a model with a run_utils function. 

The easiest way to make a new main script is by copying a simple one, e.g. swissmetro_paper_run.py. Then, change the data_manager import, filepaths, choices number, betas number and extra features number. If your own data manager is done correctly, selected models will compile, train and save. 

## Post Processing and Visualization

Current Post Processing scripts require trained models obtainable by successfully running the main '_run.py' scripts. They will get test set likelihoods, hessian estimations of parameters, beta values etc.. and save it all in a pickled dictionnary. 

Current Visualization scripts have the required dict files to show results used in the paper. 

## Dataset
### Swissmetro

Bierlaire, M., Axhausen, K., & Abay, G. (2001, March). *The acceptance of modal innovation: The case of Swissmetro.* In Proceedings of the 1st Swiss Transportation Research Conference.

Code on Utility functions inspired from [Biogeme examples](http://biogeme.epfl.ch/examples_swissmetro.html)

[Swissmetro official page](https://swissmetro.ch/)

## Our Paper Reference

Sifringer, B., Lurkin, V., & Alahi, A. (2018). *Let Me Not Lie: Learning MultiNomial Logit.* arXiv preprint arXiv:1812.09747.
