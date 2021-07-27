import tensorflow as tf

# Calculate gradient on trainnable variables

w = tf.Variable(1.0)
b = tf.Variable(0.5)

x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])

with tf.GradientTape() as tape:
    z = tf.add(tf.multiply(w,x), b)
    loss = tf.reduce_sum(tf.square(y-z))

dloss_dw = tape.gradient(loss, w)
tf.print('dL/dw: ', dloss_dw)


# calculate gradient on  untraindable values
with tf.GradientTape() as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w,x), b)
    loss = tf.reduce_sum(tf.square(y-z))

dloss_dx = tape.gradient(loss, x)
tf.print('dL/dx = ' , dloss_dx)


# Save the resources for many camputation gradient
with tf.GradientTape(persistent=True) as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dw = tape.gradient(loss, w)
tf.print('dL/dw: ', dloss_dw)
dloss_db = tape.gradient(loss, b)
tf.print('dL/db: ', dloss_db)

# Optimizer with keras for nerual model
w = tf.Variable(1.0)
b = tf.Variable(0.5)

optimizer = tf.keras.optimizers.SGD()
optimizer.apply_gradients(zip([dloss_dw, dloss_db], [w,b]))
tf.print('Updated weight: ', w)
tf.print('Updated bias: ', b)

