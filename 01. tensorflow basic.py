import tensorflow as tf


#constant() 함수로 노드 1개만 있는 그래프를 만든다
#edge는 없으며, node는 "Hello, TensorFlow!!" 라는 상수만 있는 연산 그래프이다.

#현재는 유니코드, 세션 런을 하면 바이트 스트링, 아스키코드로 변환됨 !!
hello = tf.constant("Hello, TensorFlow!!")
print(hello)



#계산 그래프를 실행하기 위해서는
#session을 만들고
#run() 함수를 사용하여 우리가 만들어놓은 operation(node)을 실행한다.
#출력에서 문자열 앞에 있는 b 는 바이트 문자열을 의미함.


sess = tf.Session()

print(sess.run(hello))
print("--------------------------------------")





#Computitional Graph

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
node3 = tf.add(node1, node2) # node1 + node 2

print("--------------------------------------")






#그냥 출력하면 그 결과는 ?
#결과값이 나오는 것이 아니라 노드의 정보가 나옴!
print("그냥 출력했을 때 : 노드의 정보")
print("node1 : ", node1, "node2 : ", node2)
print("node3 :", node3)
print("--------------------------------------")



#결과값이 나오게 하려면 !
print("결과값이 나오게 하려면?")
sess_2 = tf.Session()
print("sees.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run([node3]))
print("--------------------------------------")



#Placeholder
#adder_node 를 실행시킬때, 텐서플로우가 노드에 플레이스홀더에게 값을 넘겨 주는 것이 feed_dict
#array로 넘겨주는 것도 가능 !
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b : 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))



