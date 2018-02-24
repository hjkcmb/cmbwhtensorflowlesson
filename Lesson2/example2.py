import tensorflow as tf

x=tf.constant([[1,2],[3,4]])
y=tf.constant([[5,6],[7,8]])
s=tf.matmul(x,y)


# with tf.Session() as sess:
#     print(sess.run(s))

# sess = tf.Session()
# print(sess.run(s))
# sess.close()