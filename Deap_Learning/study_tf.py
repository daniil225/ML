import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(precision=3)

a = np.array([1,2,3], dtype=np.int32)
t_a = tf.convert_to_tensor(a)
print(t_a)

tf.random.set_seed(1)
t = tf.random.uniform((6,))

# Расшепление тензоров (общий размер должен быть кратным указанному количеству)
t_splits = tf.split(t, num_or_size_splits=3)
print([item.numpy() for item in t_splits])

t = tf.random.uniform((5,))
t_splits = tf.split(t, num_or_size_splits=[3,2])
print([item.numpy() for item in t_splits])


A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A,B], axis=0)
print(C.numpy())


B = tf.zeros((3,))
C = tf.stack([A,B], axis=1)
print(C.numpy())



