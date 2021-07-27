import tensorflow as tf

#Create simple nrual model

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

# late creation of variables
model.build(input_shape=(None, 4))
print(model.summary())

# print model variables
for v in model.variables:
    print('{:20s}'.format(v.name), v.trainable, v.shape)


# Create model
# 1 layers init weights and bias
# 2 layes init with regularization l1
model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Dense(
        units=16,
        activation=tf.keras.activations.relu,
        kernel_initializer= tf.keras.initializers.glorot_uniform(),
        bias_initializer= tf.keras.initializers.Constant(2.0)
    ))
model.add(
    tf.keras.layers.Dense(
        units=32,
        activation=tf.keras.activations.sigmoid,
        kernel_regularizer=tf.keras.regularizers.l1
    ))

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics= [tf.keras.metrics.Accuracy(),
              tf.keras.metrics.Precision(),
              tf.keras.metrics.Recall()])




