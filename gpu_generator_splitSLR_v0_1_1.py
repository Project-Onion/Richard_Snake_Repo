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
import itertools

import random
import os

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
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 52, 52, 3])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([7, 7, 3, 32]) #feature size 7x7 to have 46x46 image after convolution
        b_conv1 = bias_variable([32]) #32 feature maps - arbitrary - can change
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #uses max function (instead of sigmoid function)
        # h_conv1 = tf.nn.relu(conv2d(h_fc0_flat, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1) #now size will be 23x23x32

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64]) #feature size 5x5 to have 19x19  image after convolution
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
         h_pool2 = max_pool_2x2(h_conv2) #now size will be 9x9x64

    # Third convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv3'):
         W_conv3 = weight_variable([3, 3, 64, 128]) #feature size 3x3 to have 8x8 image after convolution
         b_conv3 = bias_variable([128])
         h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 10x10x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 128,4096]) #4096 = first power of 2 larger than 2500 (=50x50), also (8*8*128 = 8192)/2
        b_fc1 = bias_variable([4096])

        # h_pool2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * 64])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'): #maybe add dropout to other layers aswell?
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 4096 features to 3 classes, one for each direction
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([4096, 3])
        b_fc2 = bias_variable([3])

        # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

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

"""
def batchGenerator(batchData, mini_batch_size):
    for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
        batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        batch = [batchData[i][batchStartIndex:batchEndIndex] for i in range(2)]
        yield [\
            [np.array([[pixel if (pixel == 1 or pixel == 2) else 0, pixel if (pixel == 4) else 0, pixel if (pixel == 5 or pixel == 6) else 0] for pixel in image]).flatten() for image in batch[0]], \
            # [[pixel if (pixel == 4) else 0 for pixel in image] for image in batch[0]], \
            # [[pixel if (pixel == 5 or pixel == 6) else 0 for pixel in image] for image in batch[0]]], \
            batch[1]]

def batchGenerator(batchData, mini_batch_size): #our snake different color
    for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
        batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        batch = [batchData[i][batchStartIndex:batchEndIndex] for i in range(2)]
        yield [\
            [np.array([\
                [pixel if (pixel == 1 or pixel == 2) else 0 for pixel in board],\
                [pixel if (pixel == 4 or pixel == 3) else 0 for pixel in board],\
                [pixel if (pixel == 5 or pixel == 6) else 0 for pixel in board]]).flatten() for board in batch[0]], \
            batch[1]]
"""
def batchGenerator(batchData, mini_batch_size): #border different color
    for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
        batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        batch = [batchData[i][batchStartIndex:batchEndIndex] for i in range(2)]
        yield [\
            [np.array([\
                [pixel if (pixel == 1 or pixel == 2 or pixel == 3) else 0 for pixel in board],\
                [pixel if (pixel == 4 or pixel == 5) else 0 for pixel in board],\
                [pixel if (pixel == 6 or pixel == 7) else 0 for pixel in board]\
                ]).flatten() for board in batch[0]], \
            batch[1]]


def main(_):
    print("\n\n --- STARTING CODE ---")
    print("------------------------")

    startTime = time.time()

    amountOfMiniBatchFilesToTrain = 50
    amountOfMiniBatchFilesToValidate = 1
    amountOfMiniBatchFilesToTest = 15 #was 2
    starting_learning_rate = 5*1e-4 #5*1e-4  #was 1e-3
    mini_batch_size = 500   #was 500
    numEpochs = 20 #400
    dataFileNumber = 14 #was 3, then 5
    innerFolder = ""
    keep_prob_start = 0.5
    biasRatio = 1
    biasDecayFreq = 1
    biasRatioLimit = 4

    print("dataFileNumber: " + str(dataFileNumber))
    print("amountOfMiniBatchFilesToTrain: " + str(amountOfMiniBatchFilesToTrain))
    print("amountOfMiniBatchFilesToValidate: " + str(amountOfMiniBatchFilesToValidate))
    print("amountOfMiniBatchFilesToTest: " + str(amountOfMiniBatchFilesToTest))
    print ("starting learning rate: " + str(starting_learning_rate))
    print ("mini batch size: "+str(mini_batch_size))
    print ("epochs to be trained: " +str(numEpochs))
    print ("keep_prob_start: " + str(keep_prob_start))
    print ("starting bias ratio: " + str(biasRatio))
    print ("bias decay frequency: " + str(biasDecayFreq))
    print ("bias ratio limit: " + str(biasRatioLimit))
#    sys.stdout.flush()

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 1000, 0.98, staircase=True)
    confusion = np.zeros([3,3])
    tensorGenerator = tf.data.Dataset.from_generator(batchGenerator, tf.float32, tf.TensorShape([None, 2704*3, 3]))

    with tf.device('/gpu:0'):

        # Create the model
        # x = tf.placeholder(tf.float32, [None, 2704*3], name="x")

        # Define loss and optimizer
        # y_ = tf.placeholder(tf.float32, [None, 3], name="y")

        tensorBatchData = tf.placeholder(tf.float32, [None, 2704, 3], name="tensorBatchData")
        currSample = tensorGenerator.make_one_shot_iterator().get_next()

        x = tf.variable(currSample[0], name="x")
        y_ = tf.variable(currSample[1], name="y")

        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # Build the graph for the deep net
        # y_conv, keep_prob, regularizer = deepnn(x) # with l2 regularization
        y_conv, keep_prob= deepnn(x, keep_prob)

        tf.argmax(y_conv, 1, output_type=tf.int32, name="result_argmax")

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=y_conv)
        # beta = 0.01 #lambda

        cross_entropy = tf.reduce_mean(cross_entropy)# + beta*regularizer)

        with tf.name_scope('adam_optimizer'): #adam replaces gradient decent
#            learning_rate = tf.placeholder(tf.float32, name="learning_rate") #1e-3
#            learning_rate = starting_learning_rate #tf.train.exponential_decay(starting_learning_rate, global_step, 100, 0.96, staircase=True)
            # train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
#            print ("adam with learning rate: " + str(learning_rate))
#            sys.stdout.flush()
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction, name="accuracy1")
        confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1), num_classes=3, name="confusion1")
	    # confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1), num_classes=3, name="confusion1")
#        print("Variables initialized ...")
    
    sys.stdout.flush()
    # confusion = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y_conv, 1), num_classes=3)#, name="confusion")
    fileLocation = "/mnt/snake/snakeNN/snakeNN_data"+ str(dataFileNumber) +"/"

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    sys.stdout.flush()
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    print("\nstarting session")
    # with tf.Session() as sess:
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        validationBatchDataStraight = [np.load(fileLocation + "validationData/" + 'validationDataBoardsStraight' + str(1) + ".npy"),
                     np.load(fileLocation + "validationData/" + 'validationDataMovesStraight' + str(1) + ".npy")]
        validationBatchDataLeft = [np.load(fileLocation + "validationData/"+ 'validationDataBoardsLeft' + str(1) + ".npy"),
                     np.load(fileLocation + "validationData/" + 'validationDataMovesLeft' + str(1) + ".npy")]
        validationBatchDataRight = [np.load(fileLocation + "validationData/"+ 'validationDataBoardsRight' + str(1) + ".npy"),                     np.load(fileLocation + "validationData/"  + 'validationDataMovesRight' + str(1) + ".npy")]
        
        validationDataShuffle = np.array(range(len(validationBatchDataStraight[0]) + len(validationBatchDataLeft[0]) + len(validationBatchDataRight[0])))
        np.random.shuffle(validationDataShuffle)
        validationBatchData = [np.concatenate((validationBatchDataStraight[0] , validationBatchDataLeft[0] , validationBatchDataRight[0]))[validationDataShuffle] , np.concatenate((validationBatchDataStraight[1] , validationBatchDataLeft[1] , validationBatchDataRight[1]))[validationDataShuffle]]
        validationBatchDataBiased = [np.load("/mnt/snake/snakeNN/snakeNN_data11/validationData/"+ innerFolder + 'validationDataBoards' + str(1) + ".npy"),
                     np.load("/mnt/snake/snakeNN/snakeNN_data11/validationData/" + innerFolder + 'validationDataMoves' + str(1) + ".npy")]
        # final_confusion = np.zeros([3,3])
        print ("variables initialized, starting training...\n")
#        for i in range(1,amountOfMiniBatchFilesToTrain + 1):
        for epoch in range(numEpochs):
            if(epoch%biasDecayFreq == 0 and epoch !=0 and biasRatio <= biasRatioLimit and biasRatio>=1):
                if(epoch%4==1):
                    biasRatio = 2
                elif(epoch%4==2):
                    biasRatio = 1
                elif(epoch%4==3):
                    biasRatio = 3
                elif(epoch%4==3):
                    biasRatio = 4

                print("bias ratio changed to: " + str(biasRatio))
#		learning_rate = learning_rate/1.5

            straightAllFiles = sorted(os.listdir(fileLocation + "trainingData/"+ "straight/"))
            leftAllFiles = sorted(os.listdir(fileLocation + "trainingData/"+ "left/"))
            rightAllFiles = sorted(os.listdir(fileLocation + "trainingData/"+ "right/"))
            
            straightAllBoardFiles = straightAllFiles[:int(len(straightAllFiles)/2)]
            straightAllMoveFiles = straightAllFiles[int(len(straightAllFiles)/2):]
            leftAllBoardFiles = leftAllFiles[:int(len(leftAllFiles)/2)]
            leftAllMoveFiles = leftAllFiles[int(len(leftAllFiles)/2):]
            rightAllBoardFiles = rightAllFiles[:int(len(rightAllFiles)/2)]
            rightAllMoveFiles = rightAllFiles[int(len(rightAllFiles)/2):]

            straightFileShuffle = np.array(range(len(straightAllBoardFiles)))
            np.random.shuffle(straightFileShuffle)
            straightAllBoardFiles = np.array(straightAllBoardFiles)[straightFileShuffle]
            straightAllMoveFiles = np.array(straightAllMoveFiles)[straightFileShuffle]
            straightAllBoardFiles = straightAllBoardFiles[:amountOfMiniBatchFilesToTrain]
            straightAllMoveFiles = straightAllMoveFiles[:amountOfMiniBatchFilesToTrain]

            leftFileShuffle = np.array(range(len(leftAllBoardFiles)))
            np.random.shuffle(leftFileShuffle)
            leftAllBoardFiles = np.array(leftAllBoardFiles)[leftFileShuffle]
            leftAllMoveFiles = np.array(leftAllMoveFiles)[leftFileShuffle]
            leftAllBoardFiles = leftAllBoardFiles[:amountOfMiniBatchFilesToTrain]
            leftAllMoveFiles = leftAllMoveFiles[:amountOfMiniBatchFilesToTrain]

            rightFileShuffle = np.array(range(len(rightAllBoardFiles)))
            np.random.shuffle(rightFileShuffle)
            rightAllBoardFiles = np.array(rightAllBoardFiles)[rightFileShuffle]
            rightAllMoveFiles = np.array(rightAllMoveFiles)[rightFileShuffle]
            rightAllBoardFiles = rightAllBoardFiles[:amountOfMiniBatchFilesToTrain]
            rightAllMoveFiles = rightAllMoveFiles[:amountOfMiniBatchFilesToTrain]

            straightAllBoardFiles = np.array_split(straightAllBoardFiles, biasRatio)
            straightAllMoveFiles = np.array_split(straightAllMoveFiles, biasRatio)
            #straightAllBoardFiles[-1] = np.concatenate((straightAllBoardFiles[-1], straightAllBoardFiles[:(biasRatio-len(straightAllBoardFiles)%biasRatio)%biasRatio]))
            #straightAllMoveFiles[-1] = np.concatenate((straightAllMoveFiles[-1], straightAllMoveFiles[:(biasRatio-len(straightAllMoveFiles)%biasRatio)%biasRatio]))
            for i,_ in enumerate(straightAllBoardFiles):
                if straightAllBoardFiles[i].shape != straightAllBoardFiles[0].shape:
                    straightAllBoardFiles[i] = np.append(straightAllBoardFiles[i], random.choice(straightAllBoardFiles[i]))
            
            if (min(len(leftAllBoardFiles),len(rightAllBoardFiles)) < len(straightAllBoardFiles[0])):
                leftAllBoardFiles = np.tile(leftAllBoardFiles , (int(len(straightAllBoardFiles[0])/len(leftAllBoardFiles)) + 1))
                leftAllMoveFiles = np.tile(leftAllMoveFiles , (int(len(straightAllMoveFiles[0])/len(leftAllMoveFiles)) + 1))
                rightAllBoardFiles = np.tile(rightAllBoardFiles , (int(len(straightAllBoardFiles[0])/len(rightAllBoardFiles)) + 1))
                rightAllMoveFiles = np.tile(rightAllMoveFiles , (int(len(straightAllMoveFiles[0])/len(rightAllMoveFiles)) + 1))
            elif (min(len(leftAllBoardFiles),len(rightAllBoardFiles)) > len(straightAllBoardFiles[0])):
                straightAllBoardFiles = np.tile(straightAllBoardFiles , (int(max(len(leftAllBoardFiles),len(rightAllBoardFiles))/len(straightAllBoardFiles)) + 1))
                straightAllMoveFiles = np.tile(straightAllMoveFiles , (int(max(len(leftAllMoveFiles),len(rightAllMoveFiles))/len(straightAllMoveFiles)) + 1)) 

            print("straights in first group: " + str(len(straightAllBoardFiles[0])) + "\t straights in last group: " + str(len(straightAllBoardFiles[-1])))
            print("amount of straights: " + str(len(straightAllBoardFiles)*len(straightAllBoardFiles[0])) + "\t amount of left: " + str(len(leftAllBoardFiles)) + "\t amount of rights: " + str(len(rightAllBoardFiles)))
 
            global_step_start = global_step.eval()
            for i,((straightBoardFile, straightMoveFile),(leftBoardFile, leftMoveFile),(rightBoardFile, rightMoveFile)) in\
                enumerate(zip(zip(zip(*straightAllBoardFiles), zip(*straightAllMoveFiles)),\
                zip(leftAllBoardFiles, leftAllMoveFiles),\
                zip(rightAllBoardFiles, rightAllMoveFiles))):
            #if i>70:
                #starting_learning_rate = 5*1e-5
                loadDataTime = time.time()
                batchDataStraight = [\
                                        np.concatenate([np.load(fileLocation + "trainingData/" + "straight/" + boardFile) for boardFile in straightBoardFile]),\
                                        np.concatenate([np.load(fileLocation + "trainingData/" + "straight/" + moveFile) for moveFile in straightMoveFile])]
                
                batchDataLeft = [np.load(fileLocation + "trainingData/"+ "left/" + leftBoardFile),
                             np.load(fileLocation + "trainingData/" + "left/" + leftMoveFile)]
                batchDataRight = [np.load(fileLocation + "trainingData/"+ "right/" + rightBoardFile),
                             np.load(fileLocation + "trainingData/" + "right/" + rightMoveFile)]

                dataShuffle = np.array(range(len(batchDataStraight[0]) + len(batchDataLeft[0]) + len(batchDataRight[0])))
                np.random.shuffle(dataShuffle)

#                print("aaaaaaaaaaaaa" + str(batchDataStraight[1].shape) + " " + str(batchDataLeft[1].shape) + " " +str(batchDataRight[1].shape))
                batchData = [np.concatenate((batchDataStraight[0] , batchDataLeft[0] , batchDataRight[0]))[dataShuffle] , np.concatenate((batchDataStraight[1] , batchDataLeft[1] , batchDataRight[1]))[dataShuffle]]
#            for epoch in range(numEpochs):
                #rearrange = np.array(range(len(batchData[0])))
                #np.random.shuffle(rearrange)
                print("minibatch file: " + str(i) + " epoch " + str(epoch) + " started training.\tglobal step is: " + str(global_step.eval()) + "\tlearning rate is: " + str(learning_rate.eval()) + "\ttime passed: " + str(time.time() - startTime))
                sys.stdout.flush()	
                # miniBatchGenerator = batchGenerator([batchData[0][rearrange],batchData[1][rearrange]], mini_batch_size)
                loadDataTime = time.time() - loadDataTime
                trainStartTime = time.time()
                gpuTotalTime = 0
                train_step.run(feed_dict={tensorBatchData: batchData,
                                          keep_prob: keep_prob_start})
#                 for miniBatch in batchGenerator(batchData, mini_batch_size):
#                     gpuStartTime = time.time()
#                     train_step.run(
#                         feed_dict={x: miniBatch[0],
#                                    y_: miniBatch[1],
#                                    keep_prob: keep_prob_start})#,
# #                                   learning_rate: starting_learning_rate})
#                     gpuTotalTime = gpuTotalTime + (time.time() - gpuStartTime)
#                writer = tf.summary.FileWriter("/mnt/snake/snakeNN/snakeNN_code/tensorBoard/1")
#                writer.add_graph(sess.graph)
#                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

                    #global_step = global_step + 1
            #print("minibatch file: " + str(i) + " started validation. time passed: "+ str(time.time()-startTime))
                print ("minibatch file: " + str(i) + " epoch " + str(epoch) + "\tload data time: "  +str(loadDataTime) + "\ttotal train time: " + str(time.time()-trainStartTime) + "\tGPU train time: " + str(gpuTotalTime)) 
                if (global_step.eval()-global_step_start)%(((20000/mini_batch_size)*(biasRatio+2))*15) == 0 or epoch == numEpochs-1:
                    print("global step: " + str(global_step.eval()) + " started validation on SLR. time passed: "+ str(time.time()-startTime))
                    sys.stdout.flush()
                    sumOfValidations = 0
                    amountOfValidations = 0
                    final_confusion = np.zeros([3,3])
            # validationMiniBatchGenerator = batchGenerator(validationBatchData, mini_batch_size)
                    for miniBatch in batchGenerator(validationBatchData, mini_batch_size):
                        validate_accuracy = accuracy.eval(feed_dict={
                            x: miniBatch[0],
                            y_: miniBatch[1],
                            keep_prob: 1.0})#,
 #                   learning_rate: starting_learning_rate})
                        final_confusion = final_confusion + confusion.eval(feed_dict={
                            x: miniBatch[0],
                            y_: miniBatch[1],
                            keep_prob: 1.0})
                # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
                        sumOfValidations = sumOfValidations + validate_accuracy
                        amountOfValidations = amountOfValidations + 1
                #print (validate_accuracy)

            #print("minibatch file "+ str(i+1) + "/" +str(amountOfMiniBatchFilesToTrain) +  " validation: "  + str(sumOfValidations/amountOfValidations))
            #print("minibatch file " +str(i+1)+ " confusion matrix:\n" + str(final_confusion))
                    print("global step "+ str(global_step.eval()) +  " validation on SLR data: "  + str(sumOfValidations/amountOfValidations))
                    print("global step " +str(global_step.eval())+ " confusion matrix on SLR data:\n" + str(final_confusion))
                    sys.stdout.flush()

                    print("global step: " + str(global_step.eval()) + " started validation on biased data. time passed: "+ str(time.time()-startTime))
                    sys.stdout.flush()
                    sumOfValidations = 0
                    amountOfValidations = 0
                    final_confusion = np.zeros([3,3])
            # validationMiniBatchGenerator = batchGenerator(validationBatchData, mini_batch_size)
                    for miniBatch in batchGenerator(validationBatchDataBiased, mini_batch_size):
                        validate_accuracy = accuracy.eval(feed_dict={
                            x: miniBatch[0],
                            y_: miniBatch[1],
                            keep_prob: 1.0})#,
 #                   learning_rate: starting_learning_rate})
                        final_confusion = final_confusion + confusion.eval(feed_dict={
                            x: miniBatch[0],
                            y_: miniBatch[1],
                            keep_prob: 1.0})
                # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
                        sumOfValidations = sumOfValidations + validate_accuracy
                        amountOfValidations = amountOfValidations + 1
                #print (validate_accuracy)


            #print("minibatch file "+ str(i+1) + "/" +str(amountOfMiniBatchFilesToTrain) +  " validation: "  + str(sumOfValidations/amountOfValidations))
            #print("minibatch file " +str(i+1)+ " confusion matrix:\n" + str(final_confusion))
                    print("global step "+ str(global_step.eval()) +  " validation on biased data: "  + str(sumOfValidations/amountOfValidations))
                    print("global step " +str(global_step.eval())+ " confusion matrix on biased data:\n" + str(final_confusion))
                    sys.stdout.flush()
            
                    now = datetime.datetime.now()
                    save_path = saver.save(sess, '../models/output_snake_model_' +now.strftime("%Y%m%d_%H%M%S"),global_step=global_step)
                    print("Model saved in file: %s" % save_path)

        trainEndTime = time.time()
        print ("training and validation ended. \t time it took: " + str(trainEndTime - startTime))
        print ("starting testing...")
        sys.stdout.flush()
        sumOfTests = 0
        amountOfTests = 0
        for i in range(1, amountOfMiniBatchFilesToTest + 1):
            batchData = [np.load(fileLocation + "testData/" + innerFolder + 'testDataBoards' + str(i) + ".npy"),
                         np.load(fileLocation + "testData/" + innerFolder + 'testDataMoves' + str(i) + ".npy")]
#            amountOfTests = 0
#             testMiniBatchGenerator = batchGenerator(batchData, mini_batch_size)
            for miniBatch in batchGenerator(batchData, mini_batch_size):
                test_accuracy = accuracy.eval(feed_dict={
                    x: miniBatch[0],
                    y_: miniBatch[1],
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
	
        #saver = tf.train.Saver()
        # export_path = "/home/student/Desktop/saved_models/model" + str(FLAGS.model_version)#+".ckpt"
   	 #save_path = saver.save(sess, export_path)
        now = datetime.datetime.now()
        save_path = saver.save(sess, '../models/output_snake_model_final'+now.strftime("%Y%m%d_%H%M%S"))
        print("Model saved in file: %s" % save_path)
        sys.stdout.flush()

        writer = tf.summery.FileWriter("/mnt/snake/snakeNN/snakeNN_code/tensorBoard/1")
        writer.add_graph(sess.graph)


if __name__ == '__main__':

    tf.app.run(main=main)
