import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]


#tensorflow가 사용하는 variable, 텐서플로우가 학습하는 과정에서 변경시키는 변수!
#w와 b의 값을 모름으로 random 한 값으로 설정해줌

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b    #tensor



# Build graph using TF oprations

#cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#reduce_mean이란 ?
# t = [1. , 2. , 3. , 4.]
#tf.reduce_mean(t)  ==> 2.5
#평균내주는 것 !


#Minimize
#train을 실행시켜야 cost를 최소화 할 수 있다.
#cost를 최소화 시킨다는 것 == 예측값과 실제값의 차이를 최소화 시키는 것

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# Run/update graph and get result

#Launch the graph in a session

sess = tf.Session()
#그래프 내에 있는 전역변수 초기화
sess.run(tf.global_variables_initializer())


#Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(cost), sess.run(W), sess.run(b))


#텐서플로우의 전체적인 구조

'''
* train()함수는 cost()함수와 연결되어있다.
* cost()함수는 hypothesisH(x)와 연결되어있다.
* train을 실행 시킨다는 것은 그래프(train - cost - hypothesis < weight, bias
 를 만든 다음에 training 하는 것
* train()함수 실행 : 학습을 시킨다는 것
    --> 그래프를 따라 들어가 W와 b에 값을 저장한다는 것을 의미
'''