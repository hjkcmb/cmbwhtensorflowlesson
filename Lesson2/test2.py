import tensorflow as tf

x = tf.placeholder(tf.float32, shape=None)
y = tf.placeholder(tf.float32, shape=None)
z = x + y

with tf.Session() as sess:
    result = sess.run(z, feed_dict={x: 1, y: 2})
    print(result)
