
import tensorflow as tf
print "VERSION", tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
print "VERSION", tf.__version__sess 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
