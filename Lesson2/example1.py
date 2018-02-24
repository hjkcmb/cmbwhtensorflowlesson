import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

plt.ion()   # something about plotting

for step in range(201):
    _, w ,b = sess.run([train, Weights, biases])
    if step % 10 == 0:
        y = x_data * w + b
        print(step, w, b)
        plt.cla()
        plt.scatter(x_data,y_data)
        plt.plot(x_data, y, 'r-', lw=5)
        plt.pause(0.1)

plt.ioff()
plt.show()
