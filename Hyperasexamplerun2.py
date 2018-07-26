from __future__ import print_function

import numpy as np
import h5py
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import LeakyReLU, ELU
from keras import optimizers

from hyperas import optim
from hyperas.distributions import choice, uniform

from workingwithpython import *

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = x_train.reshape(60000, 784)
    #x_test = x_test.reshape(10000, 784)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    #nb_classes = 10
    #y_train = np_utils.to_categorical(y_train, nb_classes)
    #y_test = np_utils.to_categorical(y_test, nb_classes)
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from macros_AWS import scale_x
    data_directory = '/home/rice/jmc32/Gridsearch_Data/'
    data_sample = 'PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m_redo.npy'
    scaler = 'robust'
    totalset = np.load(data_directory + data_sample)
    dataset, testset = train_test_split(totalset, test_size = 0.01)
    # Split into input (X) and output (Y) variables
    x_train_prescale = dataset[:,1:]
    y_train = dataset[:,0]
    x_test_prescale = testset[:,1:]
    y_test = testset[:,0]
    # Scale
    print(y_train.shape)
    print(y_test.shape)
    #print(numpy.matrix(y_train))
    x_train, x_test = scale_x(x_train_prescale, x_test_prescale, scaler)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_test = min_max_scaler.fit_transform(x_train)
    #y_test = min_max_scaler.fit_transform(y_train)
    #x_train = min_max_scaler.fit_transform(x_train)
    #y_train = min_max_scaler.fit_transform(y_train)
    #x_test=x_test.reshape((900000, 7))
    #y_test=y_test.reshape((y_test.shape[0], 7))
    #x_train=x_train.reshape((x_train.shape[0], 7))
    #y_train=y_train.reshape((y_train.shape[0],  7))
    
    print(x_train.shape)
    print(x_test.shape)
    #y_train= to_categorical(y_train)
    #y_test= to_categorical(y_test)
    #x_train= to_categorical(x_train)
    #x_test= to_categorical(x_test)
    
    return x_train, y_train, x_test, y_test


#def makemorelayers(numlayers,neurons):
#   c=keras.layers.LeakyReLU(alpha=0.3)
#   d=model.add(Dense(neurons))
#   for val in range(0,numlayers):
    #   c
    #   d



def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense({{choice(range(2500,3001))}}, input_dim=7))
    model.add({{choice([LeakyReLU(alpha=.01),Activation('relu'),ELU(alpha=1)])}})
    model.add(Dense({{choice(range(2000,2501))}}))
    model.add({{choice([ LeakyReLU(alpha=.01),Activation('relu'),ELU(alpha=1)])}})
    model.add(Dense({{choice(range(1500,2001))}}))
    model.add({{choice([ LeakyReLU(alpha=.01),Activation('relu'),ELU(alpha=1)])}})
    model.add(Dense({{choice(range(1000,1501))}}))
    model.add({{choice([ LeakyReLU(alpha=.01),Activation('relu'),ELU(alpha=1)])}})
    model.add(Dense({{choice(range(500,1001))}}))
    model.add({{choice([ LeakyReLU(alpha=.01),Activation('relu'),ELU(alpha=1)])}})
    model.add(Dense({{choice(range(1,501))}}))
    model.add({{choice([ LeakyReLU(alpha=.01),Activation('relu'),ELU(alpha=1)])}})
    model.add(Dense(1))
  
    model.add(Activation('sigmoid'))
    adam= optimizers.Adam(lr=.001, beta_1=.99, beta_2=.999, epsilon=.0000001, decay=.0001) 
  
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], 
optimizer= 'adam')

    
    model.fit(x_train, y_train,
              batch_size={{choice(range(5000,6001))}},
              epochs={{choice(range(100,301))}},
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    #model_predictions = best_model.predict(Y_test)
    #outfile_predict = open('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/predictions/model_class_predictions_1m_kclassify.txt', 'w')
    #outfile_truth = open('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/predictions/model_class_true_1m_kclassify.txt', 'w')
    #numpy.savetxt(outfile_predict, model_predictions)
    #numpy.savetxt(outfile_truth, Y_test)
    #plot_ROC(y_test,model_predictions,1,show_toggle = True, save_toggle=True)
    best_model.save('/scratch/rice/datatest1.h5')
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
