import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt;
import numpy as np

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

a = tf.Variable(1,name="a")
b = tf.Variable(2,name="b")
f = a + b;
init = tf.global_variables_initializer()
with tf.Session() as Session:
     init.run()
     print(f.eval())


def display_sample(num):
    print(mnist.train.labels(num))
    label = mnist.train.labels(num).argmax(0)

images = mnist.train.images[0].reshape((1,784))
for i in range(1,500):
       images = np.concatenate((images,mnist.train.images[i].reshape([1,784])))

plt.imshow(images,cmap=plt.get_cmap('gray_r'))
##plt.show()

input_images = tf.placeholder(tf.float32 , shape=[None,784])
target_labels = tf.placeholder(tf.float32 , shape=[None,10])

hidden_nodes = 512;
input_weights = tf.Variable(tf.truncated_normal([784,hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes,10]))
hidden_biases = tf.Variable(tf.zeros([10]))

input_layer = tf.matmul(input_images,input_weights)
hidden_layer = tf.nn.relu(input_layer+input_biases)
digit_weights = tf.matmul(hidden_layer,hidden_weights) + hidden_biases


loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit=digit_weights,labels=target_labels))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)
correct_predictions = tf.equal()