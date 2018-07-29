#cost function의 시각화

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = x * W

#cost function
cost = tf.reduce_mean(tf.square(hypothesis - y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#cost 함수를 그래프로 출력하기 위한 변수 설정
#그래프를 그리기 위해 w값과 cost값을 저장할 리스트 생성
W_val = []
cost_val = []

#-3 ~ 5까지 0.1 의 간격으로 움직임
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict = {W : feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)


plt.plot(W_val, cost_val)
plt.show()


