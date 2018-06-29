from __future__ import print_function

import numpy
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import to_categorical

from hyperas import optim
from hyperas.distributions import choice, uniform

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
    from macros_AWS import scale_x
    data_directory = '/home/rice/jmc32/Gridsearch_Data/'
    data_sample = 'PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m_redo.npy'
    scaler = 'maxabs'
    totalset = numpy.load(data_directory + data_sample)
    dataset, testset = train_test_split(totalset, test_size = 0.1)
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
    print(x_train.shape)
    print(x_test.shape)
    #y_train= to_categorical(y_train)
    #y_test= to_categorical(y_test)
    #x_train= to_categorical(x_train)
    #x_test= to_categorical(x_test)
    return x_train, y_train, x_test, y_test


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
    model.add(Dense({{choice([100, 150, 50, 6])}}, input_dim=7))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))    
    model.add(Dense({{choice([100,150,50,6])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(.5)])}})
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation({{choice(['softmax','sigmoid'])}}))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop','adam','sgd'])}})

    model.fit(x_train, y_train,
              batch_size={{choice(range(50,200,10))}},
              epochs={{choice(range(1,50))}},
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
