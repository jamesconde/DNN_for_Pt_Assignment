import numpy as np

predicts = np.loadtxt('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/predictions/model_class_predictions_1m_kclassify.txt')

#totaldata = np.genfromtxt('/storage1/users/eb8/Gridsearch_Data/PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m.csv', delimiter = ',')
truth = np.loadtxt('/home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/predictions/model_class_true_1m_kclassify.txt')

it = 0
true_neg = 0
true_pos = 0
false_neg = 0
false_pos = 0
missed = 0

for val in predicts:
	if (int(val) == truth[it] and truth[it] == 0):
		true_neg += 1
	elif (int(val) == truth[it] and truth[it] == 1):
		true_pos += 1
	elif (int(val) != truth[it] and truth[it] == 0):
		false_pos += 1
	elif (int(val) != truth[it] and truth[it] == 1):
		false_neg += 1
	else:
		missed += 1	
	it += 1

file = open('./logFile_1m_kclassify.txt', 'w')

print('True negative total is ', true_neg, 'out of', it)
print('True positive total is ', true_pos)
print('False negative total is ', false_neg)
print('False positive total is ', false_pos)
print('Any missed ones? ', missed)

file.write('True negative total is %d out of %d \n' %(true_neg,it))
file.write('True positive total is %d \n' %true_pos)
file.write('False negative total is %d \n' %false_neg)
file.write('False positive total is %d \n' %false_pos)
file.write('Any missed ones? %d \n' %missed)
file.write('Efficiency = %f \n' %(float(true_pos)/float(true_pos+false_neg)))
file.write('Rate = %f \n' %(float(true_neg+false_neg)/it))
file.write('Accuracy = %f \n' %(float(true_neg+true_pos)/it))
file.close()
