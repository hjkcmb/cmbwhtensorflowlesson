# MNIST分类问题,
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import  matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
# 定义softmax回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
y = tf.matmul(x, W) + b


y_ = tf.placeholder(tf.float32, [None, 10])
#增加隐藏层
# neural network layers
l1 = tf.layers.dense(x, 50, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 10)

l2 = tf.layers.dense(l1, 50, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l2, 10)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(loss)


sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())

print(y_)
print(output)

for step in range(1000):
    # train and net output
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run([train_op, loss, output], {x: batch_xs ,y_: batch_ys})

# 评估训练好的模型
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 计算模型准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
