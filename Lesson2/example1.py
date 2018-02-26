import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#固定随机种子
tf.set_random_seed(1)
np.random.seed(1)

# 创建数据
x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x_data.shape)
y_data = np.power(x_data, 2) + noise

# plot data
plt.scatter(x_data, y_data)
plt.show()

x=tf.placeholder(tf.float32,x_data.shape)
y=tf.placeholder(tf.float32,y_data.shape)

# 神经网络层
l1 = tf.layers.dense(x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
#optimizer=tf.train.AdamOptimizer()
train = optimizer.minimize(loss)
#初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
plt.ion()
#训练
for step in range(100):
    _, l, pred = sess.run([train, loss, output], {x: x_data, y: y_data})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
