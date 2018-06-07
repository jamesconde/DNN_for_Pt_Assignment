import os
import numpy
import pandas
import time
import sys 					#to write out stuff printed to screen
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,LearningRateScheduler 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import maxabs_scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Bring in and set global variables must be done before macros!
import globals
globals.init()

# Bring in external macros
import macros_AWS
from macros_AWS import *


# Create directory structure
if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
if not os.path.exists('./saved_models/'):
    os.makedirs('./saved_models/')
if not os.path.exists('./logs/'):
    os.makedirs('./logs/')
if not os.path.exists('./predictions/'):
    os.makedirs('./predictions/')
if not os.path.exists('./roc_info/'):
    os.makedirs('./roc_info/')


#Set the start time for the entire process
overall_start_time = time.time()

#Open the log file
#!!! Always check/change your log file name and location
file = open('./logs/logFile_gridsearch_1m_small_redo.txt', 'w')

# Define Constants
#!!! Always check which data you're using
data_directory = '/storage1/users/eb8/Gridsearch_Data/'
data_sample = 'PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m_redo.npy'
#test_data_sample = 'not1000_test.npy'
scaler = 'maxabs'
feature = 7
number_of_loops = 1							#Total number of loops, is incremented later for functions who's index start at 0
number_of_epochs = 1							#Just what it says, number of epochs never re-indexed
set_batch_size = 10000							#Select batch size

# Fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Log Constants
file.write('--------------------------------\n')
file.write('    Definitions of Constants    \n')
file.write('--------------------------------\n')
file.write('Directory: %s\n'        % data_directory)
file.write('Data: %s\n'    % data_sample)
#file.write('Test data: %s\n'        % test_data_sample)
file.write('Seed value: %d\n'       % seed)
file.write('Feature number: %d\n'   % feature)
file.write('Number of loops: %d\n'  % number_of_loops)
file.write('Number of epochs: %d\n' % number_of_epochs)
file.write('Batch Size: %d\n'       % set_batch_size)
file.write('********************************\n')
file.write('********************************\n')



# Load UCI data
totalset = numpy.load(data_directory + data_sample) 
#Load data from numpy array
#testset = numpy.load(data_directory + test_data_sample)                      	#Load data from numpy array
dataset, testset = train_test_split(totalset, test_size = 0.1)

# Split into input (X) and output (Y) variables
X_train_prescale = dataset[:,1:]
Y_train = dataset[:,0]
X_test_prescale = testset[:,1:]
Y_test = testset[:,0]


# Scale
X_train, X_test = scale_x(X_train_prescale, X_test_prescale, scaler)


## Pull the input layer dimension
globals.input_scale = X_train.shape[1]


loop_start_time = time.time()

model = KerasClassifier(build_fn=create_model, nb_epoch=1, batch_size=10000, verbose=0) 
#model = KerasClassifier(build_fn=create_model_neurons, nb_epoch=1, batch_size=10000, verbose=0)
#model = KerasClassifier(build_fn=create_model_opt, nb_epoch=1, batch_size=10000, verbose=0)
#model = KerasClassifier(build_fn=create_model_activation, nb_epoch=1, batch_size=10000, verbose=0)


# define the grid search parameters
#!!! Change these to vary the grid search
batch_size = [1,100,1000]
nb_epoch = [1,2,5]
#batch_size = [1]
#nb_epoch = [1, 5, 10, 50]
param_grid = dict(batch_size=batch_size, nb_epoch=nb_epoch)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)

### define the grid search parameters
##neurons = [1,5,10,50,100,500,1000,5000,10000]
#neurons = [1,5,10]
#param_grid = dict(neurons=neurons)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X_train, Y_train)

## define the grid search parameters
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#param_grid = dict(optimizer=optimizer)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X_train, Y_train)

## define the grid search parameters
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#param_grid = dict(activation=activation)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X_train, Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
file.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
   	print("%f (%f) with: %r" % (mean, stdev, param))
	file.write("%f (%f) with: %r\n" % (mean, stdev, param))


overall_end_time = time.time()
overall_elapsed_time = overall_end_time-overall_start_time
print "Done"
file.write('Elapsed Time = %d seconds' % overall_elapsed_time)
file.write('X Feature Count: %d\n'     % globals.input_scale) 
file.close()
