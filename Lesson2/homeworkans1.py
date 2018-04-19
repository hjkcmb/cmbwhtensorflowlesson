import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子
tf.set_random_seed(1)
np.random.seed(1)

# 训练数据
x = np.linspace(-1, 1, 20)[:, np.newaxis]
y = x + 0.3 * np.random.randn(20)[:, np.newaxis]

# 测试数据
test_x = x.copy()
test_y = test_x + 0.3 * np.random.randn(20)[:, np.newaxis]

# 显示图表
plt.scatter(x, y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 传入数据

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, [None, 1], name='tf_x')
    tf_y = tf.placeholder(tf.float32, [None, 1], name='tf_y')
    tf_is_training = tf.placeholder(tf.bool, None, name='tf_is_training')

# 神经网络，2层+Dropout
with tf.variable_scope('Net'):
    d1 = tf.layers.dense(tf_x, 100, tf.nn.relu, name='layer_hide_1')
    d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training, name='dropout_1')  # drop out 50% of inputs
    d2 = tf.layers.dense(d1, 100, tf.nn.relu, name='layer_hide_2')
    d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training, name='dropout_2')  # drop out 50% of inputs
    d_out = tf.layers.dense(d2, 1, name='layer_out')

# 定义loss和optimizer
d_loss = tf.losses.mean_squared_error(tf_y, d_out, scope='loss')
tf.summary.scalar('loss', d_loss)

with tf.variable_scope('Train'):
    d_train = tf.train.AdamOptimizer(0.001).minimize(d_loss)

# 初始化参数

sess = tf.Session()
sess.run(tf.global_variables_initializer())
plt.ion()  # something about plotting

writer = tf.summary.FileWriter('./log', sess.graph)  # write to file
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
merge_op = tf.summary.merge_all()  # operation to merge all summary

# 训练
for t in range(500):
    _, result = sess.run([d_train, merge_op], {tf_x: x, tf_y: y, tf_is_training: True})
    writer.add_summary(result, t)
    if t % 20 == 0:
        plt.cla()
        d_loss_, d_out_ = sess.run(
            [d_loss, d_out], {tf_x: test_x, tf_y: test_y, tf_is_training: False}
            # test, set is_training=False
        )
        plt.scatter(x, y, c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x, d_out_, 'b-', lw=3, label='dropout(50%)')
        plt.text(0, 0.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 16, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.pause(0.1)

plt.ioff()
plt.show()
