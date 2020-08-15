import os
import random
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data

class MyModel(object):
  def train(self):
    parser = argparser.ArgumentParser()
    parser.add_argument('--learning_rate', required=False, type=float, default=0.01)
    parser.add_argument('--dropout_rate', required=False, type=float, default=0.2)
    args = parser.parse_args()
    
    mnist = input_data.read_data_set("MNIST_data", one_hot=True)
    
    learning_rate = args.learning_rate
    print("### learning_rate: ", learning_rate)
    training_epochs = 5
    batch_size = 100
    
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    
    keep_probe = tf.placeholder(tf.float32)
    
    W1 = tf.get_variable("W1", shape=[784, 512],
                        initializer=tf.contrib.layers.xavier_initializer)
    b1 = tf.get_variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    
    W2 = tf.get_variable("W2", shape=[512, 10],
                        initializer=tf.contrib.layers.xavier_initializer)
    b2 = tf.get_variable(tf.random_normal([10]))
    
    hypothesis = tf.matmul((L1, W2) + b2)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, label=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variable_initializer())
    
    # train by model
    for epoch in range(training_epochs) :
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch) :
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # keep_prob: Network의 70%를 유지해서 학습함
            feed_dict = { X: batch_xs, Y: batch_ys, keep_prob: args.dropout_rate}
            C, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += C /total_batch
        
        print('Epoch: ', '%04d' % (epoch+1), 'cost=', '{:.9}'.format(avg_cost))
        
        correct_predition =tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        
        # validation accuracy
        print('Validation-accuracy=' + str(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1 })))
        
    print('Learning Finished!')
    

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
    
    for epoch in range(5) :
        r = random.randint(0, mnist.test.num_examples - 1)
        print("\nTest Image -> ", sess.run(tf.argmax(mnist.test.labels[r:r], 1)))
        
        # plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Grey', interpolation='nearest')
        # plt.show()
        print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1], keep_prob: 1}))

    
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils

        # DOCKER_REGISTRY = 'kubeflow-registry.default.svc.cluster.local:30000'
        DOCKER_REGISTRY = 'index.docker.io/insoopark'
        fairing.config.set_builder(
            'append',
            image_name='katib-job',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-cpu',
            registry=DOCKER_REGISTRY, 
            push=True)
        # cpu 2, memory 4GiB
        fairing.config.set_deployer('job',
                                    namespace='admin',
                                    pod_spec_mutators=[
                                        k8s_utils.get_resource_mutator(cpu=2,
                                                                       memory=4)]
         
                                   )
        # python3         
        # fairing.config.set_preprocessor('python', input_files=[__file__])
        fairing.config.run()
    else:
        remote_train = MyModel()
        remote_train.train()
