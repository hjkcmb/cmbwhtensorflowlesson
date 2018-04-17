import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子
tf.set_random_seed(1)
np.random.seed(1)

# 创建数据
x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x_data.shape)
y_data = np.power(x_data, 2) + noise

with tf.variable_scope('Inputs'):
    x = tf.placeholder(tf.float32, x_data.shape, name='x')
    y = tf.placeholder(tf.float32, y_data.shape, name='y')

# 神经网络层
with tf.variable_scope('Net'):
    l1 = tf.layers.dense(x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(y, output, scope='loss')
tf.summary.scalar('loss', loss)

with tf.variable_scope('Train'):
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./log', sess.graph)  # write to file
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
merge_op = tf.summary.merge_all()  # operation to merge all summary

# 训练
for step in range(100):
    _, result = sess.run([train_op, merge_op], {x: x_data, y: y_data})
    writer.add_summary(result, step)
