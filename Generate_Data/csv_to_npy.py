import numpy as np

arry = np.genfromtxt('/storage1/users/eb8/Gridsearch_Data/PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m_redo.csv', delimiter=',')

np.save('/storage1/users/eb8/Gridsearch_Data/PtRegression_for_DNN_Vars_MODE_15_noBitCompr_RPC_1m_redo.npy',arry)
