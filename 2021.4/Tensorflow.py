# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 21:36:48 2021

@author: zcg54
"""

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
coefficients = np.array([[1.], [-20.], [100.]])

w = tf.Variable(0., dtype=tf.float32)
X = tf.compat.v1.placeholder(tf.float32, [3, 1])
# cost = tf.add(w**2, tf.add(tf.multiply(-10., w), 25.))
# cost = w**2 - 10 * w + 25
cost = X[0][0] * w**2 + X[1][0] * w + X[2][0]

tf.compat.v1.disable_eager_execution()
train = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as session:
    session.run(init)
    for i in range(10000):
        session.run(train, feed_dict={X:coefficients})
    # print(session.run(w))
    print(session.run(w))


np.random.randn(3,1)



'''-----------------------------------------------------------------------'''
X = tf.constant([[1., 2., 3.], [4., 5., 6.]])
y = tf.constant([[10.], [20.]])

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
            )
    
    def call(self, input):
        output = self.dense(input)
        return output
    
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables) ## 命名好像也是从零开始计数的，有待以后查证
