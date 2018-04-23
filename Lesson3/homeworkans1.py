# MNIST分类问题,
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# 定义softmax回归模型
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

layer1 = tf.layers.dense(x, 100, tf.nn.relu)
layer2 = tf.layers.dense(layer1, 100, tf.nn.relu)
out = tf.layers.dense(layer2, 10)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y)
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 定义会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 训练1000次,每次随机抓取100个
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
# 评估训练好的模型
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 计算模型准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,}))
