#참고로 이건 넘나 복잡해 사용안함 ^^


import tensorflow as tf



#훈련용 데이터 입력
x1_data = [73., 93., 89., 96., 73. ]
x2_data = [80., 88., 91., 98., 66. ]
x3_data = [75., 93., 90., 100., 70. ]
y_data = [152., 185., 180., 196., 142 ]

#placeholder로 텐서 자료 입력
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

#w와 b는 난수를 발생하여 조정하는 값이기 때문에 변수로 설정 !
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight12')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')


#가설식은 식 (1)처럼 입력
hypothesis = x1*w1 + x2*w2 + x3*w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#cost함수의 극솟값을 구하기 위해 경사하강법 최적화 메서드 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x1 : x1_data, x2 : x2_data, x3 : x3_data, Y : y_data})

    if step % 10 == 0 :
        print(step, "Cost : ", cost_val, "\nPrediction:\n", hy_val)