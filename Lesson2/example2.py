import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子
tf.set_random_seed(1)
np.random.seed(1)

# 创建数据
x_data = np.linspace(-1, 1, 30)[:, np.newaxis]
y_data = np.power(x_data, 2) + np.random.normal(0, 0.2, size=x_data.shape)

# test data
x_test = np.linspace(-1, 1, 30)[:, np.newaxis]
y_test = np.power(x_test, 2) + np.random.normal(0, 0.2, size=x_test.shape)

# show data
plt.scatter(x_data, y_data, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(x_test, y_test, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.show()

# tf placeholders
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

# overfitting net
o1 = tf.layers.dense(tf_x, 100, tf.nn.relu)
o2 = tf.layers.dense(o1, 100, tf.nn.relu)
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(tf_y, o_out)
o_train = tf.train.AdamOptimizer(0.01).minimize(o_loss)

# dropout net
d1 = tf.layers.dense(tf_x, 100, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)  # drop out 50% of inputs
d2 = tf.layers.dense(d1, 100, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)  # drop out 50% of inputs
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(0.01).minimize(d_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()  # something about plotting

for t in range(500):
    sess.run([o_train, d_train], {tf_x: x_data, tf_y: y_data, tf_is_training: True})  # train, set is_training=True
    if t % 10 == 0:
        # plotting
        plt.cla()
        o_loss_, d_loss_, o_out_, d_out_ = sess.run(
            [o_loss, d_loss, o_out, d_out], {tf_x: x_test, tf_y: y_test, tf_is_training: False}
            # test, set is_training=False
        )
        plt.scatter(x_data, y_data, c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(x_test, y_test, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(x_test, o_out_, 'r-', lw=3, label='overfitting');
        plt.plot(x_test, d_out_, 'b-', lw=3, label='dropout(50%)')
        plt.text(0, 0.25, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 16, 'color': 'red'})
        plt.text(0, 0.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 16, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.pause(0.1)

plt.ioff()
plt.show()
