import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
plt.scatter(x_data, y_data)
plt.show()

Weigths = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype=tf.float32)
biases = tf.Variable(tf.zeros([1]))
y = x_data * Weigths + biases

loss = tf.losses.mean_squared_error(y_data, y)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
plt.ion()

for step in range(500):
    _, W, b = sess.run([train, Weigths, biases])
    if step % 20 == 0:
        print(step,W, b)
        y=x_data*W+b
        plt.cla()
        plt.scatter(x_data,y_data)
        plt.plot(x_data,y,'r-',lw=5)
        plt.pause(0.5)

plt.ioff()
plt.show()