import numpy as np
from macros_AWS import *
from sklearn import preprocessing
orgdata=np.load("testset7_21_2018.csv")
newdata=robust_scale(orgdata)
np.savetext("datasetafterrobustscaling.csv",newdata,delimiter=",")
