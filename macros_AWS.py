import os
import numpy
import pandas
import time
import sys                                      #to write out stuff printed to screen
from hyperopt import Trials, STATUS_OK,tpe
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from keras.models import Sequential
#from keras.initializers import RandomNomral
#from keras import initializers
#from keras.initializers import Zeros
from keras.layers import Dense, Dropout, Activation
#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.layers import LeakyReLU, ELU

from sklearn.preprocessing import *
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import maxabs_scale, robust_scale, minmax_scale
from sklearn.model_selection import GridSearchCV

from hyperas import optim
from hyperas.distributions import choice, uniform

import globals


def scale_x(X_train_prescale, X_test_prescale, scaler):
	#X_train_prescale=numpy.ones((900000,7), dtype=numpy.int)
	#X_test_prescale=numpy.ones((100000,7), dtype=numpy.int)
	#scaler='robust'
	
	if scaler=='none':
		X_train = X_train_prescale
		X_test  = X_test_prescale
		print 'No prescale\n'
	elif scaler=='maxabs':
		X_train = maxabs_scale(X_train_prescale, axis=0, copy=True)
		X_test  = maxabs_scale(X_test_prescale, axis=0, copy=True)
		print 'MaxAbs\n'	
	elif scaler=='robust_scale':
		X_train = robust_scale(X_train_prescale)
		X_test  = robust_scale(X_test_prescale)
		print 'Robust Scale\n'
	elif scaler == 'minmax':
                X_train = minmax_scale(X_train_prescale)
                X_test  = minmax_scale(X_test_prescale)
	elif scaler == 'robust':
		X_train= robust_scale(X_train_prescale, axis=0, with_centering=True, with_scaling=True, quantile_range=(1, 99), copy=True)
                X_test = robust_scale(X_test_prescale, axis=0, with_centering=True, with_scaling=True, quantile_range=(1, 99), copy=True)
	return  X_train, X_test

#	'elif scaler == 'robustailor':
#	X_train= robust_scale(X_train_prescale[:,1:4], axis=0, with_centering=True, with_scaling=True, quantile_range=(25, 75), copy=True)	
#		X_test = robust_scale(X_test_prescale[:,1:4], axis=0, with_centering=True, with_scaling=True, quantile_range=(25, 75), copy=True)	
#		x1=X_train_prescale[:,4:8] 
		#print X_test_prescale.shape
		#print x1.shape
		#print X_train.shape
		#print X_test_prescale[:,0]
#		xn1=  (X_test_prescale[:,0]).reshape(100000,1)
#		qt = QuantileTransformer(output_distribution='normal')
 #               x00=qt.fit_transform(x1)

		#print x00.shape
#		x2=X_test_prescale[:,4:8]
		#print x2.shape
		#print X_test.shape
#		xn2=  (X_train_prescale[:,0]).reshape(900000,1)
#                qt = QuantileTransformer(output_distribution='normal')
#                x11=qt.fit_transform(x2)
		#print x00.shape
		#print x11.shape
		#print xn1.shape
		#print xn2.shape
#		X_test=numpy.concatenate((xn1,X_test,x11), axis=1)
#		X_train=numpy.concatenate((xn2,X_train,x00), axis=1)
#		print X_test.shape
#		print X_train.shape
#	return X_train, X_test'


#cale_X(x_train_prescale,X_test_prescale,scaler)

#from TestandTrain1NetConfig1AI.py import x_train_prescale, y_train,x_test_prescale,y_test

#def create_model():
    	#model = Sequential()
    	#model.add(Dense(512, input_shape=(784,)))
    	#model.add(Activation('relu'))
    	#model.add(Dropout({{uniform(0, 1)}}))
    	#model.add(Dense({{choice([256, 512, 1024])}}))
    	#model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    	#model.add(Dropout({{uniform(0, 1)}}))

    	# If we choose 'four', add an additional fourth layer
    	#if {{choice(['three', 'four'])}} == 'four':
        #	model.add(Dense(100))
        #	model.add({{choice([Dropout(0.5), Activation('linear')])}})
        #	model.add(Activation('relu'))

    	#	model.add(Dense(10))
    	#	model.add(Activation('softmax'))

    	#model.compile(loss='categorical_crossentropy',
        #          optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
        #          metrics=['accuracy'])

    	#model.fit(x_train, y_train,
        #      	batch_size={{choice([64, 128])}},
        #      	nb_epoch=1,
        #      	verbose=2,
        #      	validation_data=(x_test, y_test))
    	#score, acc = model.evaluate(x_test, y_test, verbose=0)
    	#print('Test accuracy:', acc)
    	#return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    	#if __name__ == '__main__':
       	#	trials = Trials()
       	#	best_run, best_model = optim.minimize(model=model,
        #                                  data=data,
        #                                  algo=tpe.suggest,
        #                                  max_evals=5,
        #                                  trials=trials)
    	#for trial in trials:
        #	print(trial)
    	#x_train, y_train, x_test, y_test = data()
    	#print("Evalutation of best performing model:")
    	#print(best_model.evaluate(x_test, y_test))

#	input_model = Sequential()
#	n1=1000
#	n2=10
	#n3=10
	#n4=100
	#n5=50
#	print "n1 is %d" %n1
#	print "n2 is %d" %n2
	#print "n3 is %d" %n3
	#print "n4 is %d" %n4
	#print "n5 is %d" %n5
	#print "n6 is 4"
 #       input_model.add(Dense(n1, input_dim=globals.input_scale, kernel_initializer='uniform', activation='relu')) #Input layer
  #      input_model.add(Dense(n2, kernel_initializer='uniform', activation='relu'))		  		      #Hidden Layer 02
        #input_model.add(Dense(n3, kernel_initializer='uniform', activation='relu'))               		      #Hidden Layer 03
        #input_model.add(Dense(n4, kernel_initializer='uniform', activation='relu'))                                  #Hidden Layer 04
	#input_model.add(Dense(n5, kernel_initializer='uniform', activation='relu'))                                  #Hidden Layer 05
       	#input_model.add(Dense(4, kernel_initializer='uniform', activation='relu'))                                  #Hidden Layer 06
	
#	input_model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))              		      #Output Layer
        #Compile model
 #       input_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #input_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#	return input_model

def create_model1():
        input_model = Sequential()
        input_model.add(Dense(100, input_dim=globals.input_scale, kernel_initializer='uniform', activation='relu')) #Input layer
        input_model.add(Dense(100, kernel_initializer='uniform', activation='relu'))		  		      #Hidden Layer 02
        input_model.add(Dense(100, kernel_initializer='uniform', activation='relu'))               		      #Hidden Layer 03
        input_model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))              		      #Output Layer
        # Compile model
        input_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #input_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return input_model


def create_model():
#For this model recommeded batch size 5434 and epochs 168
        model = Sequential()
        model.add(Dense(2865, input_dim=globals.input_scale))
	model.add(LeakyReLU(alpha=0.1)) #Input layer
        
	model.add(Dense(2075))
	model.add(Activation('relu'))		  		      #Hidden Layer 02
        model.add(Dense(1905)) 
	model.add(ELU(alpha=1.0))               		      #Hidden Layer 03
	model.add(Dense(1500))
        model.add(ELU(alpha=1.0))

	model.add(Dense(761))
        model.add(ELU(alpha=1.0))

	model.add(Dense(75))
        model.add(Activation('relu'))

	model.add(Dense(1, activation='sigmoid'))              		      #Output Layer
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #input_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

def create_model_UCI_27D_default():
	#random_normal = RandomNormal(mean=0.0, stddev=0.1, seed=8675309)
        input_model = Sequential()
        #input_model.add(Dense(500, input_dim=globals.input_scale, init='uniform', activation='relu')) #Input layer
	input_model.add(Dense(500, input_dim=globals.input_scale, init='uniform', activation='relu', weights=[numpy.random.normal(0,0.1,(globals.input_scale, 500)),numpy.random.normal(0,0.1,500)])) #Input layer
        input_model.add(Dense(500, init='uniform', activation='relu', weights=[numpy.random.normal(0,0.1,(500, 500)),numpy.random.normal(0,0.1,500)]))                                #Hidden Layer 02
        input_model.add(Dense(500, init='uniform', activation='relu', weights=[numpy.random.normal(0,0.1,(500, 500)),numpy.random.normal(0,0.1,500)]))                                #Hidden Layer 03
	input_model.add(Dense(500, init='uniform', activation='relu', weights=[numpy.random.normal(0,0.1,(500, 500)),numpy.random.normal(0,0.1,500)]))                                 #Hidden Layer 04
	input_model.add(Dense(500, init='uniform', activation='relu', weights=[numpy.random.normal(0,0.1,(500, 500)),numpy.random.normal(0,0.1,500)]))                                 #Hidden Layer 05
        input_model.add(Dense(1, init='uniform', activation='sigmoid',  weights=[numpy.random.normal(0,0.1,(500, 1)),numpy.random.normal(0,0.1,1)]))                                #Output Layer
        # Compile model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
        input_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        return input_model

def create_model_UCI_1D_default():
        input_model = Sequential()
        input_model.add(Dense(100, input_dim=globals.input_scale, init='uniform', activation='sigmoid')) #Input layer
        input_model.add(Dense(100, init='uniform', activation='sigmoid'))                                #Hidden Layer 02
        input_model.add(Dense(100, init='uniform', activation='sigmoid'))                                #Hidden Layer 03
        input_model.add(Dense(1, init='uniform', activation='sigmoid'))                                  #Output Layer
        # Compile model
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	input_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        return input_model

def create_model_neurons(neurons = 40): 
        input_model = Sequential()
        input_model.add(Dense(neurons, input_dim=globals.input_scale, init='uniform', activation='relu')) #Input layer
        input_model.add(Dense(neurons, init='uniform', activation='relu'))              		  #Hidden Layer 02
        input_model.add(Dense(neurons, init='uniform', activation='relu'))               		  #Hidden Layer 03
        input_model.add(Dense(1, init='uniform', activation='sigmoid'))					  #Output Layer
	# Compile model
	input_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return input_model



def create_model_activation(activation='tanh'):
        input_model = Sequential()
        input_model.add(Dense(100, input_dim=globals.input_scale, init='uniform', activation=activation)) #Input layer
        input_model.add(Dense(100, init='uniform', activation=activation))               		  #Hidden Layer 02
        input_model.add(Dense(100, init='uniform', activation=activation))               		  #Hidden Layer 03
        input_model.add(Dense(1, init='uniform', activation='sigmoid'))					  #Output Layer
        # Compile model
        input_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return input_model



def create_model_opt(optimizer='adam'):
        input_model = Sequential()
        input_model.add(Dense(100, input_dim=globals.input_scale, init='uniform', activation='relu')) #Input layer
        input_model.add(Dense(100, init='uniform', activation='relu'))               		      #Hidden Layer 02
        input_model.add(Dense(100, init='uniform', activation='relu'))               		      #Hidden Layer 03
        input_model.add(Dense(1, init='uniform', activation='sigmoid'))				      #Output Layer
        # Compile model
        input_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return input_model

def plot_accuracy(history_arg, x, show_toggle, save_toggle):
        plt.plot(history_arg.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if show_toggle == True:
                plt.show()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_accuracy_truemass_%02d.png' % x)
        plt.close()
        return

def plot_loss(history_arg, x, show_toggle, save_toggle):
        plt.plot(history_arg.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if show_toggle == True:
                plt.show()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_loss_truemass_%02d.png' % x)
        plt.close()
        return

def plot_ppredictredictions(feature_arg, x_test_data, predictions, x, show_toggle, save_toggle):
        # Setup plot
        bins_array = numpy.arange(-5,5,0.1)
        plt.xlabel('Feature %d' % feature_arg)
        plt.ylabel('Events')
        plt.title('Distro of ttWW Reconstructed Mass')

        # Plot predictions
        plt.hist(x_test_data[numpy.nonzero(predictions.T==0)[1]][:, feature_arg-1], bins_array, alpha=0.3, color='r', label=['bkg'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==1500))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='y', label=['1500'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==1250))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='c', label=['1250'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==1000))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='g', label=['1000'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]==750))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='b', label=['750'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,27]<500))[1])][:,feature_arg-1], bins_array, alpha=0.3, color='m', label=['500'])
        plt.legend()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_predictions_truemass_%02d.png' % x)
        if show_toggle == True:
                plt.show()
        plt.close()

def plot_predictions_1D(feature_arg, x_test_data, predictions, x, show_toggle, save_toggle):
        # Setup plot
        bins_array = numpy.arange(-5,5,0.1)
        plt.xlabel('Feature %d' % feature_arg)
        plt.ylabel('Events')
        plt.title('Distro of ttWW Reconstructed Mass')

	x_test_data.ndim 
 	x_test_data.shape

        # Plot predictions
        plt.hist(x_test_data[numpy.nonzero(predictions.T==0)[1]][:,0], bins_array, alpha=0.3, color='r', label=['bkg'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,1]==1500))[1])][:,0], bins_array, alpha=0.3, color='y', label=['1500'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,1]==1250))[1])][:,0], bins_array, alpha=0.3, color='c', label=['1250'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,1]==1000))[1])][:,0], bins_array, alpha=0.3, color='g', label=['1000'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,1]==750))[1])][:,0], bins_array, alpha=0.3, color='b', label=['750'])
        plt.hist(x_test_data[(numpy.nonzero((predictions.T) & (x_test_data[:,1]<500))[1])][:,0], bins_array, alpha=0.3, color='m', label=['500'])
        plt.legend()
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_predictions_truemass_%02d.png' % x)
        if show_toggle == True:
                plt.show()
        plt.close()

def plot_ROC(y_test_data, predictions, x, show_toggle, save_toggle):
        #Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test_data, predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        if show_toggle == True:
                plt.show()
        print('\n')
        print('AUC: %f' % roc_auc)
        if save_toggle == True:
                plt.savefig('./plots/27D_not1000_ROC_truemass_%02d.png' % x)
        numpy.save('./roc_info/fpr_not1000_%02d' %x, fpr)
        numpy.save('./roc_info/tpr_not1000_%02d' %x, tpr)
        plt.close()
	return roc_auc 

