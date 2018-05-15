# MNIST分类问题,
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
BATCH_SIZE = 100
LR = 0.001
drop_hide = 0.5

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])
tf_is_training = tf.placeholder(tf.bool, None)
image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层 过滤器大小为5*5, 当前层深度为1，过滤器的深度为32
# 第二层最大池化层 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
# 第三层卷积层 过滤器大小为5*5，当前层深度为32，过滤器深度为64
# 第四层最大池化层 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
# 全连接层1输出【1024】
# 为了减少过拟合，加入Dropout层
# 全连接层2输出[10]






# softmax分类和交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

# 用平均值来统计测试准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy=tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 开始训练
for step in range(1000):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_op, {x: batch_x, y: batch_y, tf_is_training: True})
    if step % 5 == 0:
        accuracy_ = sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels, tf_is_training: False})
        print('Step:', step, '| test accuracy: ', accuracy_)
