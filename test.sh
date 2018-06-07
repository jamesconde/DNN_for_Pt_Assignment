#!/bin/bash
source activate tensorflow
python -c 'import tensorflow as tf; print(tf.__version__)'
python -c 'import keras; print(keras.__version__)'
