import tensorflow as tf


#instance data
x_data = [[73., 80., 75], [93., 88., 93], [98., 91., 90.],
          [96., 98., 100.], [73., 66., 70.]]

y_data =[[152.],[185.],[180.],[196.],[142]]

#placeholder를 사용하여 텐서 자료를 입력
#x의 차원은 5*3이지만 인스턴스의 갯수는 얼마인지 모르기 때문에 None으로 설정
X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(20001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={X : x_data, Y : y_data})

    if step % 5000 == 0 :
        print(step, "Cost : ", cost_val, "\nPrediction:\n", hy_val)
