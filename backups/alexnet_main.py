# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.
This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014
Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.
Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)
@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import time
import sys
import datetime
import tensorflow as tf
import tempfile
import numpy as np

tf.app.flags.DEFINE_integer('training_iteration', 1,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/student/Desktop/', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      activation_fn=nn_ops.relu,
      biases_initializer=init_ops.constant_initializer(0.1),
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope([layers.conv2d], padding='SAME'):
      with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=3,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=[end_points_collection]):
      inputs_image = tf.reshape(inputs, [-1, 50, 50, 1])
      net = layers.conv2d(
          inputs_image, 64, [11, 11], 1, padding='VALID', scope='conv1')
      net = layers_lib.max_pool2d(net, [3, 3], 1, scope='pool1')
      net = layers.conv2d(net, 192, [5, 5], scope='conv2')
      net = layers_lib.max_pool2d(net, [3, 3], 1, scope='pool2')
      net = layers.conv2d(net, 384, [3, 3], scope='conv3')
      net = layers.conv2d(net, 384, [3, 3], scope='conv4')
      net = layers.conv2d(net, 256, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [3, 3], 1, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with arg_scope(
          [layers.conv2d],
          weights_initializer=trunc_normal(0.005),
          biases_initializer=init_ops.constant_initializer(0.1)):
        net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
        net = layers_lib.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
        net = layers_lib.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        net = layers.conv2d(
            net,
            num_classes, [3, 3],
            activation_fn=None,
            normalizer_fn=None,
            biases_initializer=init_ops.zeros_initializer(),
            scope='fc8')

      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


alexnet_v2.default_image_size = 50

def main(_):
    startTime = time.time()
    #
    # mini_batch_amount = 123123
    # amountOfMiniBatchFilesToTrain = 100
    # amountOfMiniBatchFilesToValidate = 50
    # amountOfMiniBatchFilesToTest = 50
    #
    # print ("amountOfMiniBatchFilesToTrain: " + str(amountOfMiniBatchFilesToTrain) + "\tamountOfMiniBatchFilesToValidate: " + str(amountOfMiniBatchFilesToValidate) + "\tamountOfMiniBatchFilesToTest: " + str(amountOfMiniBatchFilesToTest))

    amountOfMiniBatchFilesToTrain = 40
    amountOfMiniBatchFilesToValidate = 40
    amountOfMiniBatchFilesToTest = 20
    starting_learning_rate = 1e-3
    mini_batch_size = 500
    numEpochs = 48
    dataFileNumber = 3

    print("amountOfMiniBatchFilesToTrain: " + str(amountOfMiniBatchFilesToTrain))
    print("amountOfMiniBatchFilesToValidate: " + str(amountOfMiniBatchFilesToValidate))
    print("amountOfMiniBatchFilesToTest: " + str(amountOfMiniBatchFilesToTest))
    print("starting learning rate: " + str(starting_learning_rate))
    print("mini batch size: " + str(mini_batch_size))
    print("epochs to be trained: " + str(numEpochs))

    fileLocation = "/mnt/snake/snakeNN/snakeNN_data"+ str(dataFileNumber) +"/"

    global_step = tf.Variable(0, trainable=False)
    confusion = np.zeros([3, 3])
    with tf.device('/gpu:0'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 2500], name="x")

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 3], name="y")

        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # Build the graph for the deep net
        # y_conv, keep_prob, regularizer = deepnn(x) # with l2 regularization
        y_conv, endpoint = alexnet_v2(x, spatial_squeeze=False)

        tf.argmax(y_conv, 1, output_type=tf.int32, name="result_argmax")

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=y_conv)
        # beta = 0.01 #lambda
        cross_entropy = tf.reduce_mean(cross_entropy)# + beta*regularizer)

        with tf.name_scope('adam_optimizer'): #adam replaces gradient decent
            learning_rate = 1e-3#tf.placeholder(tf.float32, shape=[])
            # train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
            print ("adam with learning rate: " + str(learning_rate))

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1), num_classes=3)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    mini_batch_size = 50
    print ("mini batch size: "+str(mini_batch_size))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("variables initialized, starting training...\n")

        validationBatchData = [
            np.load(fileLocation + "validationData/" + 'validationDataBoards' + str(1) + ".npy"),
            np.load(fileLocation + "validationData/" + 'validationDataMoves' + str(1) + ".npy")]

        for i in range(1, amountOfMiniBatchFilesToTrain):
            batchData = [np.load(fileLocation + "trainingData/" + 'trainingDataBoards' + str(i) + ".npy"),
                         np.load(fileLocation + "trainingData/" + 'trainingDataMoves' + str(i) + ".npy")]
            for epoch in range(numEpochs):
                rearrange = np.array(range(len(batchData[0])))
                np.random.shuffle(rearrange)
                print(
                    "minibatch file: " + str(i) + " epoch " + str(epoch + 1) + " started training. time passed: " + str(
                        time.time() - startTime))
                sys.stdout.flush()
                for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
                    batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                    train_step.run(
                        feed_dict={x: batchData[0][rearrange][batchStartIndex:batchEndIndex],
                                   y_: batchData[1][rearrange][batchStartIndex:batchEndIndex],
                                   keep_prob: 0.5})  # ,
                    #                                   learning_rate: starting_learning_rate})
                    global_step = global_step + 1
            print("minibatch file: " + str(i) + " started validation. time passed: " + str(time.time() - startTime))
            sys.stdout.flush()
            sumOfValidations = 0
            # batchData = [np.load(fileLocation + "validationData/" + 'validationDataBoards' + str(i) + ".npy"),
            #              np.load(fileLocation + "validationData/" + 'validationDataMoves' + str(i) + ".npy")]
            amountOfValidations = 0
            final_confusion = np.zeros([3, 3])
            for batchStartIndex in range(0, len(validationBatchData[0]), mini_batch_size):
                batchEndIndex = batchStartIndex + min(mini_batch_size, len(validationBatchData[0]) - batchStartIndex)
                validate_accuracy = accuracy.eval(feed_dict={
                    x: validationBatchData[0][batchStartIndex: batchEndIndex],
                    y_: validationBatchData[1][batchStartIndex:batchEndIndex],
                    keep_prob: 1.0})  # ,
                #                   learning_rate: starting_learning_rate})
                final_confusion = final_confusion + confusion.eval(feed_dict={
                    x: validationBatchData[0][batchStartIndex: batchEndIndex],
                    y_: validationBatchData[1][batchStartIndex:batchEndIndex],
                    keep_prob: 1.0})
                # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
                sumOfValidations = sumOfValidations + validate_accuracy
                amountOfValidations = amountOfValidations + 1
                # print (validate_accuracy)

            print("minibatch file " + str(i + 1) + "/" + str(amountOfMiniBatchFilesToTrain) + " validation: " + str(
                sumOfValidations / amountOfValidations))
            print("minibatch file " + str(i + 1) + " confusion matrix:\n" + str(final_confusion))
            sys.stdout.flush()

        trainEndTime = time.time()
        print("training and validation ended. \t time it took: " + str(trainEndTime - startTime))
        print("starting testing...")
        sys.stdout.flush()
        sumOfTests = 0
        amountOfTests = 0
        for i in range(1, amountOfMiniBatchFilesToTest):
            batchData = [np.load(fileLocation + "testData/" + 'testDataBoards' + str(i) + ".npy"),
                         np.load(fileLocation + "testData/" + 'testDataMoves' + str(i) + ".npy")]
            #            amountOfTests = 0
            for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
                batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                test_accuracy = accuracy.eval(feed_dict={
                    x: batchData[0][batchStartIndex:batchEndIndex],
                    y_: batchData[1][batchStartIndex:batchEndIndex],
                    keep_prob: 1.0})  # ,
                #                    learning_rate: starting_learning_rate})
                # print('epoch %d, test accuracy %g' % (epoch, test_accuracy))
                sumOfTests = sumOfTests + test_accuracy
                amountOfTests = amountOfTests + 1

        print("test accuracy: " + str(sumOfTests / amountOfTests))
        sys.stdout.flush()
        # print('test accuracy %g' % accuracy.eval(feed_dict={
        #     x: np.stack(numpyCombinedTestData[0]), y_: np.stack(numpyCombinedTestData[1]), keep_prob: 1.0}))

        testEndTime = time.time()
        print("testing ended. \t time for testing: " + str(testEndTime - trainEndTime) + "\t total time: " + str(
            testEndTime - startTime))
        sys.stdout.flush()

        print("final confusion matrix:\n" + str(final_confusion))
        sys.stdout.flush()

        saver = tf.train.Saver()
        # export_path = "/home/student/Desktop/saved_models/model" + str(FLAGS.model_version)#+".ckpt"
        # save_path = saver.save(sess, export_path)
        now = datetime.datetime.now()
        save_path = saver.save(sess, 'models/output_snake_model_' + now.strftime("%Y%m%d_%H%M%S"))
        print("Model saved in file: %s" % save_path)
        sys.stdout.flush()
        # sess.run(tf.global_variables_initializer())
        # numEpochs = 10
        # for epoch in range(numEpochs):
        #
        #     print ("epoch " + str(epoch+1) + " started training. time passed: "+ str(time.time()-startTime))
        #     for i in range(0,amountOfMiniBatchFilesToTrain):
        #         batchData = [np.load(fileLocation + "trainingData/" + 'trainingDataBoards' + str(i) + ".npy"),
        #                      np.load(fileLocation + "trainingData/" + 'trainingDataMoves' + str(i) + ".npy")]
        #         rearrange = np.array(range(len(batchData[0])))
        #         np.random.shuffle(rearrange)
        #         for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
        #             batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        #             train_step.run(
        #                 feed_dict={x: batchData[0][rearrange][batchStartIndex:batchEndIndex],
        #                            y_: batchData[1][rearrange][batchStartIndex:batchEndIndex]})
        #     print("epoch " + str(epoch+1) + " started validation. time passed: "+ str(time.time()-startTime))
        #     sumOfValidations = 0
        #     for i in range(0, amountOfMiniBatchFilesToValidate):
        #         batchData = [np.load(fileLocation + "validationData/" + 'validationDataBoards' + str(i) + ".npy"),
        #                      np.load(fileLocation + "validationData/" + 'validationDataMoves' + str(i) + ".npy")]
        #         amountOfValidations = 0
        #         for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
        #             batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        #             validate_accuracy = accuracy.eval(feed_dict={
        #                 x: batchData[0][batchStartIndex: batchEndIndex],
        #                 y_: batchData[1][batchStartIndex:batchEndIndex]})
        #             # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
        #             sumOfValidations = sumOfValidations + validate_accuracy
        #             amountOfValidations = amountOfValidations + 1
        #             #print (validate_accuracy)
        #
        #     print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t epoch "+ str(epoch+1) + "/" +str(numEpochs) +  ": "  + str(sumOfValidations/amountOfValidations))
        #
        # trainEndTime = time.time()
        # print ("training and validation ended. \t time it took: " + str(trainEndTime - startTime))
        #
        # sumOfTests = 0
        # for i in range(0, amountOfMiniBatchFilesToTest):
        #     batchData = [np.load(fileLocation + "testData/" + 'testDataBoards' + str(i) + ".npy"),
        #                  np.load(fileLocation + "testData/" + 'testDataMoves' + str(i) + ".npy")]
        #     amountOfTests = 0
        #     for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
        #         batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        #         test_accuracy = accuracy.eval(feed_dict={
        #             x: batchData[0][batchStartIndex:batchEndIndex],
        #             y_: batchData[1][batchStartIndex:batchEndIndex]})
        #         # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
        #         sumOfTests = sumOfTests + test_accuracy
        #         amountOfTests = amountOfTests + 1
        #
        # print("test accuracy: " + str(sumOfTests / amountOfTests))
        # # print('test accuracy %g' % accuracy.eval(feed_dict={
        # #     x: np.stack(numpyCombinedTestData[0]), y_: np.stack(numpyCombinedTestData[1]), keep_prob: 1.0}))
        #
        # testEndTime = time.time()
        # print("testing ended. \t time for testing: " + str(testEndTime - trainEndTime) + "\t total time: "+ str(testEndTime - startTime))
        #
        #
        # export_path = "/home/student/Desktop/saved_models/model" + str(FLAGS.model_version)#+".ckpt"
        # #save_path = saver.save(sess, export_path)
        # save_path = saver.save(sess, './models/output_snake_model')
        # print("Model saved in file: %s" % save_path)


if __name__ == '__main__':

    tf.app.run(main=main)