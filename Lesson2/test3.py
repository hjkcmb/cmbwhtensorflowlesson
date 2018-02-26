import tensorflow as tf

x = tf.Variable(0)    # 变量

add = tf.add(x, 1) #加法操作
update = tf.assign(x, add)#赋值操作，将add赋值给x

with tf.Session() as sess:
    # 对所有变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update)
        print(sess.run(x))