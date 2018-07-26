import matplotlib
matplotlib.use('Agg')
import os
import numpy
import pandas
import time
import sys 					#to write out stuff printed to screen
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import h5py

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout
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

#  The point of this file is to either load a saved DNN model and then train and test it or to import the DNN model created in macros_AWS under create_model and test/train that.

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
#!!! Always check your log file name and location
file = open('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/logs/logFile_gridsearch_1m_kclassify.txt', 'w')

# Define Constants
#!!! Always double check which data you are using
data_directory = '/home/rice/jmc32/Gridsearch_Data/'
data_sample = 'PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m_redo.npy'
#test_data_sample = 'not1000_test.npy'
scaler = 'robust' #robust Setting different scales will cause different preprocessing actions to occur over the data speficied.  
feature = 7
number_of_loops = 1							#Total number iof loops, is incremented later for functions who's index start at 0
number_of_epochs = 2#417 # 292					#Just what it says, number of epochs never re-indexed
set_batch_size = 100 #4116 # 4047							#Select batch size

#print data_directory
#print data_sample

#'True positive total is ', 52398)
# Fix random seed for reproducibility
#seed = 46
#print 'Seed value is %d' % seed
#numpy.random.seed(seed)

# Log Constants
file.write('--------------------------------\n')
file.write('    Definitions of Constants    \n')
file.write('--------------------------------\n')
file.write('Directory: %s\n'        % data_directory)
file.write('Data: %s\n'    % data_sample)
#file.write('Seed value: %d\n'       % seed)
#file.write('Feature number: %d\n'   % feature)
#file.write('Number of loops: %d\n'  % number_of_loops)
file.write('Number of epochs: %d\n' % number_of_epochs)
file.write('Batch Size: %d\n'       % set_batch_size)
file.write('********************************\n')
file.write('********************************\n')



# Load data
totalset = numpy.load(data_directory + data_sample)
dataset, testset = train_test_split(totalset, test_size = 0.01)

# Split into input (X) and output (Y) variables
x_train_prescale = dataset[:,1:]
y_train = dataset[:,0]
x_test_prescale = testset[:,1:]
y_test = testset[:,0]


# Scale
x_train, x_test = scale_x(x_train_prescale, x_test_prescale, scaler)

## Pull the input layer dimension
globals.input_scale = x_train.shape[1]


loop_start_time = time.time()

print 'Using New network'

#!!! Use this to modify which model you are testing
model = KerasClassifier(build_fn=create_model, nb_epoch=number_of_epochs, batch_size=set_batch_size, verbose=0)
#model = load_model('/scratch/rice/bestonesofar12.h5')
history = model.fit(x_train,y_train, nb_epoch=number_of_epochs, batch_size = set_batch_size)

model_predictions = model.predict(x_test)

#model.model.save('/scratch/rice/modeltestsave.h5')

#model_json = model.model.to_json()
#with open("/scratch/rice/model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("/scratch/rice/model.h5")
#print("Saved model to disk")

#!!! Always check where your model results are being saved
outfile_predict = open('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/predictions/model_class_predictions_1m_kclassify.txt', 'w')
outfile_truth = open('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/predictions/model_class_true_1m_kclassify.txt', 'w')
numpy.savetxt(outfile_predict, model_predictions)
numpy.savetxt(outfile_truth, y_test)
plot_ROC(y_test,model_predictions,1,show_toggle = True, save_toggle=True) 


overall_end_time = time.time()
overall_elapsed_time = overall_end_time-overall_start_time
print "Done"
file.write('Elapsed Time = %d seconds' % overall_elapsed_time)
file.write('X Feature Count: %d\n'     % globals.input_scale) 
file.close
