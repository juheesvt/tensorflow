'''
cost 함수의 극솟값 찾기

-기울기는 미분을 이용해 찾는다 !
    오른쪽은 기울기가 +이고, 왼쪽은 -
    기울기가 +이면 왼쪽으로 움직여야하기때문에 기울기를 빼준다
    기울기가 -이면 오른쪽으로 움직여야 하기 때문에 기울기를 더해준다
-코드로 구현 !!
'''

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [2, 4, 6]


W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W*X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#미분계수 (기울기)로 경사 하강법을 사용
# W = W - learning_rate * 미분계수 ( gradient )

learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

#cost함수가 복잡할 땐 일일이 미분할 필요없이 다음 명령어 사용 !  
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
#train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict = { X : x_data, Y : y_data })
    print(step, sess.run(cost, feed_dict = { X : x_data, Y : y_data}), sess.run(W))


