import tensorflow as tf

g = tf.Graph()

with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')
    z = 2*(a-b) + c



with tf.compat.v1.Session(graph=g) as sess:
    print('Result: z = ',sess.run(z))


# TF v2

a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')
z = 2*(a-b)+c
#tf.print('Result: z = ', z)



# Loading input data TF v1.x
g = tf.Graph()
with g.as_default():
    a = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_a')
    b = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_b')
    c = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_c')
    z = 2*(a-b)+c

with tf.compat.v1.Session(graph=g) as sess:
    feed_dict = {a:1, b:2, c:3}
    #print('Result: z = ', sess.run(z, feed_dict=feed_dict))



# Load inputing data TF v2

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32)))
def comput_z(a,b,c):
    r1 = tf.subtract(a,b)
    r2 = tf.multiply(2,r1)
    z = tf.add(r2,c)
    return z

#tf.print('inputing data rang 1: ', comput_z([1],[2],[3]))


# Initialize normal Gloro
tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()
#tf.print(init(shape=(3,3)))

v = tf.Variable(init(shape=(2,3)))
#tf.print(v)


# Practice example with trainable variables and gloro initialization
class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2,3)),
                              trainable=True)
        self.w2 = tf.Variable(init(shape=(1,2)),
                              trainable=False)

m = MyModule()
print('All variables of modeule: ', [v.shape for v in m.variables])
print('Trainable Value: ', [v.shape for v in m.trainable_variables])


# Using Variable with decorated function

w = tf.Variable(tf.random.uniform((3,3)))
@tf.function
def compute_z(x):
    return  tf.matmul(w,x)

x = tf.constant([[1], [2], [3]], dtype=tf.float32)
tf.print(compute_z(x))