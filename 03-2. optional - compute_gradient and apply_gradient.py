'''
#gradient를 임의로 조정하고 싶을때
'''
import tensorflow as tf

X = [1, 2, 3]
Y = [2, 4, 6]

W = tf.Variable(5.)

hypothesis = X*W

gradient = tf.reduce_mean((W*X - Y)*X)*2

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#get gradients
gvs = optimizer.compute_gradients(cost)

#apply gradient
apply_gradients = optimizer.apply_gradients(gvs)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)