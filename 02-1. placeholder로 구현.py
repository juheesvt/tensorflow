import tensorflow as tf
'''
- placeholder() 함수로 실행할 때 변수의 값을 줄 수 있다.
- 그래프를 미리 만들어 놓고, 실행하는 단계에서 값을 주고 싶을 때 !

    1. placeholder()함수로 변수형 점을 만든다.
    2. run()함수로 그래프를 실행할 때, feed_dict 변수로 데이터를 입력한다.
'''

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x * W + b    #tensor

cost = tf.reduce_mean(tf.square(hypothesis - y))


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
            feed_dict = {x:[1,2,3],
                        y: [2,4,6]})

    if step % 20 == 0 :
        print(step, cost_val, W_val, b_val)
