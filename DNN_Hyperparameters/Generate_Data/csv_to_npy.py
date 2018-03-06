import numpy as np

arry = np.genfromtxt('PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC.csv', delimiter=',')

np.save('PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC.npy',arry)
