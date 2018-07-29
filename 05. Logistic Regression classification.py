import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# weight의 shape은 X의 변수가 2개이기 때문에 [2, 1](2행 1열)이다.
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설식
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost(loss) 함수
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# 경사 하강법
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
## 여기까지가 학습을 위한 그래프를 만들었다.

## 예측한 값을 가지고 성공했는지 실피했는지 bianry 출력해야한다.
## hypothesis를 계산하면 0과 1사이의 값이기 때문에 분류의 기준을 0.5로 설정
## 0.5보다 크면 성공, 작으면 실패

# cast() 메서드를 사용하면 결과값은 0.0 또는 1.0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# 예측값과 실제값이 같은지를 확인하여 맞을 확률을 구한다
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

## 이제 학습을 시키자.
# 그래프를 만든다.
with tf.Session() as sess:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

        if step % 2000 == 0:
            print(f"STEP = {step:06}, cost 함수값= {cost_val:1.14}")

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("\n가설식의 값 = {h},\n실제의 값 = {c},\n정확도 = {a}")
