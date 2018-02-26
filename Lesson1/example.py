import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
plt.scatter(x_data, x_data)
plt.show()
# 创建TensorFlow结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = x_data * Weights + biases

# 创建模型
loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
#optimizer=tf.train.AdamOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化参数
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
plt.ion()
# 训练
for step in range(201):
    _, W, b = sess.run([train, Weights, biases])
    if step % 20 == 0:
        y = x_data * W + b
        print(step, W, b)
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, y, 'r-', lw=5)
        plt.pause(0.5)
plt.ioff()
plt.show()
