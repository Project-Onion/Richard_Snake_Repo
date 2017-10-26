# Based off of:
# http://ischlag.github.io/2016/06/12/async-distributed-tensorflow/
# https://www.tensorflow.org/deploy/distributed
# Aswell as tensorflow mnist tutorial for experts

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time
import sys
import datetime

import random

import numpy as np

import tensorflow as tf



tf.app.flags.DEFINE_integer('training_iteration', 1,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/student/Desktop/', 'Working directory.')

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def deepnn(x, keep_prob):
    epsilon = 1e-3
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 52, 52, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):

        W_conv1 = weight_variable([5, 5, 1, 32]) #feature size 5x5 to have 46x46 image after convolution
        # b_conv1 = bias_variable([32]) #32 feature maps - arbitrary - can change
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))# + b_conv1) #uses max function (instead of sigmoid function)
        # h_conv1 = tf.nn.relu(conv2d(h_fc0_flat, W_conv1) + b_conv1)
        batch_mean1, batch_var1 = tf.nn.moments(h_conv1, [0])
        scale2 = tf.Variable(tf.ones([32]))
        beta2 = tf.Variable(tf.zeros([32]))
        BN1 = tf.nn.batch_normalization(h_conv1, batch_mean1, batch_var1, beta2, scale2, epsilon)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(BN1) #now size will be 23x23x32

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([4, 4, 32, 64]) #feature size 4x4 to have 20x20 image after convolution
        # b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))# + b_conv2)
        # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        batch_mean2, batch_var2 = tf.nn.moments(h_conv2, [0])
        scale2 = tf.Variable(tf.ones([64]))
        beta2 = tf.Variable(tf.zeros([64]))
        BN2 = tf.nn.batch_normalization(h_conv2, batch_mean2, batch_var2, beta2, scale2, epsilon)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(BN2) #now size will be 10x10x64

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 10x10x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([10 * 10 * 64, 4096]) #4096 = first power of 2 larger than 2500 (=50x50)
        b_fc1 = bias_variable([4096])

        # h_pool2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * 64])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    # with tf.name_scope('dropout'): #maybe add dropout to other layers aswell?
    #     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 4096 features to 3 classes, one for each direction
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([4096, 3])
        b_fc2 = bias_variable([3])

        # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    #regularizer = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
    # regularizer = tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)

    return y_conv, keep_prob#, regularizer

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') #VALID = no padding


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID') #VALID = no padding

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
    print("\n\n --- STARTING CODE ---")
    print("------------------------")

    startTime = time.time()

    amountOfMiniBatchFilesToTrain = 1#90
    amountOfMiniBatchFilesToValidate = 1
    amountOfMiniBatchFilesToTest = 1#30
    starting_learning_rate = 5*1e-4
    mini_batch_size = 500
    numEpochs = 1#5
    dataFileNumber = 5
    innerFolder = ""

    print("amountOfMiniBatchFilesToTrain: " + str(amountOfMiniBatchFilesToTrain))
    print("amountOfMiniBatchFilesToValidate: " + str(amountOfMiniBatchFilesToValidate))
    print("amountOfMiniBatchFilesToTest: " + str(amountOfMiniBatchFilesToTest))
    print ("starting learning rate: " + str(starting_learning_rate))
    print ("mini batch size: "+str(mini_batch_size))
    print ("epochs to be trained: " +str(numEpochs))
#    sys.stdout.flush()

    global_step = tf.Variable(0, trainable=False)
    confusion = np.zeros([3,3])
    with tf.device('/gpu:0'):

        # Create the model
        x = tf.placeholder(tf.float32, [None, 2704], name="x")

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 3], name="y")

        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # Build the graph for the deep net
        # y_conv, keep_prob, regularizer = deepnn(x) # with l2 regularization
#	tf.add(x,0,name="x_check")
	y_conv, keep_prob= deepnn(x, keep_prob)
#        tf.add(y_conv,0,name="y_conv")

        tf.argmax(y_conv, 1, output_type=tf.int32, name="result_argmax")
#        tf.argmax(y_conv, 0, output_type=tf.int32, name="result_argmax2")
#        tf.argmax(y_conv, 1, name="result_argmax3")
#        tf.argmax(y_conv, 0, name="result_argmax4")

        with tf.name_scope('loss'):
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv)

        # beta = 0.01 #lambda
	#cross_entropy_squared = (cross_entropy * cross_entropy)
        #cross_entropy_squared = tf.reduce_mean(cross_entropy_squared)# + beta*regularizer)
        cross_entropy = tf.reduce_mean(cross_entropy)# + beta*regularizer)

        with tf.name_scope('adam_optimizer'): #adam replaces gradient decent
#            learning_rate = tf.placeholder(tf.float32, name="learning_rate") #1e-3
            learning_rate = starting_learning_rate #tf.train.exponential_decay(starting_learning_rate, global_step, 100, 0.96, staircase=True)
            # train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#            print ("adam with learning rate: " + str(learning_rate))
#            sys.stdout.flush()
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

#        print("Variables initialized ...")
    
    sys.stdout.flush()
    confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1), num_classes=3)
    fileLocation = "/mnt/snake/snakeNN/snakeNN_data"+ str(dataFileNumber) +"/"

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    sys.stdout.flush()
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    print("\nstarting session")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print ("variables initialized, starting training...\n")

        validationBatchData = [np.load(fileLocation + "validationData/"+ innerFolder + 'validationDataBoards' + str(1) + ".npy"),
                         np.load(fileLocation + "validationData/"+ innerFolder + 'validationDataMoves' + str(1) + ".npy")]

        for i in range(1,amountOfMiniBatchFilesToTrain+1):
            batchData = [np.load(fileLocation + "trainingData/"+ innerFolder + 'trainingDataBoards' + str(i) + ".npy"),
                         np.load(fileLocation + "trainingData/"+ innerFolder + 'trainingDataMoves' + str(i) + ".npy")]
            for epoch in range(numEpochs):
                rearrange = np.array(range(len(batchData[0])))
                np.random.shuffle(rearrange)
                print("minibatch file: " + str(i) + " epoch " + str(epoch + 1) + " started training. time passed: " + str(time.time() - startTime))
                sys.stdout.flush()
                for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
                    batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                    train_step.run(
                        feed_dict={x: batchData[0][rearrange][batchStartIndex:batchEndIndex],
                                   y_: batchData[1][rearrange][batchStartIndex:batchEndIndex],
                                   keep_prob: 0.5})#,
#                                   learning_rate: starting_learning_rate})
                    global_step = global_step + 1
            print("minibatch file: " + str(i) + " started validation. time passed: "+ str(time.time()-startTime))
            sys.stdout.flush()
            sumOfValidations = 0
           # batchData = [np.load(fileLocation + "validationData/" + 'validationDataBoards' + str(i) + ".npy"),
           #              np.load(fileLocation + "validationData/" + 'validationDataMoves' + str(i) + ".npy")]
            amountOfValidations = 0
            final_confusion = np.zeros([3,3])
            for batchStartIndex in range(0, len(validationBatchData[0]), mini_batch_size):
                batchEndIndex = batchStartIndex + min(mini_batch_size, len(validationBatchData[0]) - batchStartIndex)
                validate_accuracy = accuracy.eval(feed_dict={
                    x: validationBatchData[0][batchStartIndex: batchEndIndex],
                    y_: validationBatchData[1][batchStartIndex:batchEndIndex],
                    keep_prob: 1.0})#,
 #                   learning_rate: starting_learning_rate})
                final_confusion = final_confusion + confusion.eval(feed_dict={
                    x: validationBatchData[0][batchStartIndex: batchEndIndex],
                    y_: validationBatchData[1][batchStartIndex:batchEndIndex],
                    keep_prob: 1.0})
                # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
                sumOfValidations = sumOfValidations + validate_accuracy
                amountOfValidations = amountOfValidations + 1
                #print (validate_accuracy)

            print("minibatch file "+ str(i+1) + "/" +str(amountOfMiniBatchFilesToTrain) +  " validation: "  + str(sumOfValidations/amountOfValidations))
            print("minibatch file " +str(i+1)+ " confusion matrix:\n" + str(final_confusion))
            sys.stdout.flush()

        trainEndTime = time.time()
        print ("training and validation ended. \t time it took: " + str(trainEndTime - startTime))
        print ("starting testing...")
        sys.stdout.flush()
        sumOfTests = 0
	amountOfTests = 0
        for i in range(50, 50 + amountOfMiniBatchFilesToTest):
            batchData = [np.load(fileLocation + "testData/"+ innerFolder + 'testDataBoards' + str(i) + ".npy"),
                         np.load(fileLocation + "testData/" + innerFolder + 'testDataMoves' + str(i) + ".npy")]
#            amountOfTests = 0
            for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
                batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                test_accuracy = accuracy.eval(feed_dict={
                    x: batchData[0][batchStartIndex:batchEndIndex],
                    y_: batchData[1][batchStartIndex:batchEndIndex],
                    keep_prob: 1.0})#,
#                    learning_rate: starting_learning_rate})
                #print('epoch %d, test accuracy %g' % (epoch, test_accuracy))
                sumOfTests = sumOfTests + test_accuracy
                amountOfTests = amountOfTests + 1

        print("test accuracy: " + str(sumOfTests / amountOfTests))
        sys.stdout.flush()
        # print('test accuracy %g' % accuracy.eval(feed_dict={
        #     x: np.stack(numpyCombinedTestData[0]), y_: np.stack(numpyCombinedTestData[1]), keep_prob: 1.0}))

        testEndTime = time.time()
        print("testing ended. \t time for testing: " + str(testEndTime - trainEndTime) + "\t total time: "+ str(testEndTime - startTime))
        sys.stdout.flush()
 	
        print("final confusion matrix:\n" + str(final_confusion))
        sys.stdout.flush()
	
        saver = tf.train.Saver()
        # export_path = "/home/student/Desktop/saved_models/model" + str(FLAGS.model_version)#+".ckpt"
   	 #save_path = saver.save(sess, export_path)
        now = datetime.datetime.now()
	save_path = saver.save(sess, 'models/output_snake_model_'+now.strftime("%Y%m%d_%H%M%S"))
        print("Model saved in file: %s" % save_path)
        sys.stdout.flush()


if __name__ == '__main__':

    tf.app.run(main=main)
