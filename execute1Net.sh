#!/bin/bash
source activate tensorflow
python /home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/GPU_Check.py
python /home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/TestandTrain1NetConfig1AI.py
python /home/rice/jmc32/DNN_for_Pt_Assignment-master/DNN_Hyperparameters/compare_truth.py

