from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import pickle
#from socket import *
import sys

#Server variables
#serverName='146.141.56.15' #'10.10.187.175'
#serverPort=12000

# Setting up connection to server
#clientSocket = socket(AF_INET, SOCK_STREAM)  # define and open client socket on client (with IPV4 and TCP)
#clientSocket.connect((serverName, serverPort))  # connecting to server


print("importing validation data file...")
fileLocation = "/mnt/snake/snakeNN/snakeNN_data5/"
innerFolder=""
validationBatchData = [np.load(fileLocation + "validationData/"+ innerFolder + 'validationDataBoards' + str(1) + ".npy"), np.load(fileLocation + "validationData/" + innerFolder + 'validationDataMoves' + str(1) + ".npy")]


def batchGenerator(batchData, mini_batch_size):
    for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
        batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
        batch = [batchData[i][batchStartIndex:batchEndIndex] for i in range(2)]
        yield batch
#        yield [\
#            [np.array([[pixel if (pixel == 1 or pixel == 2) else 0, pixel if (pixel == 4) else 0, pixel if (pixel == 5 or pixel == 6) else 0] for pixel in image]).flatten() for image in batch[0]], \
#            batch[1]]

def drawBoard(board):
    for i in range(2704):
        if board[i] == 0:
            print(" ", end=" ")
        else:
            print(board[i], end=" ")

        if (i%52==51):
            print("")

with tf.Session() as sess:

    print("importing model...")
#    saver = tf.train.import_meta_graph('/mnt/snake/snakeNN/snakeNN_code/models/good_models/SLR_sphered/20171021120723/output_snake_model_20171021_171335.meta')
#    saver.restore(sess, tf.train.latest_checkpoint('/mnt/snake/snakeNN/snakeNN_code/models/good_models/SLR_sphered/20171021120723/'))
    saver = tf.train.import_meta_graph('/mnt/snake/snakeNN/snakeNN_code/models/output_snake_model_20171026_110041.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/mnt/snake/snakeNN/snakeNN_code/models/'))

    print("setting up graph variables...")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    result_argmax = graph.get_tensor_by_name("result_argmax:0")
#    result_argmax2 = graph.get_tensor_by_name("result_argmax2:0")
#    result_argmax3 = graph.get_tensor_by_name("result_argmax3:0")
#    result_argmax4 = graph.get_tensor_by_name("result_argmax4:0")
#    y_conv = graph.get_tensor_by_name("y_conv:0")
#    x_check = graph.get_tensor_by_name("x_check:0")

#    accuracy = graph.get_tensor_by_name("accuracy:0")
#    confusion = graph.get_tensor_by_name("confusion:0")

    print("starting validation...")
    result = [0,0,0]
 #   sumOfValidations = 0
 #   amountOfValidations = 0
 #   final_confusion = np.zeros([3,3])
    for miniBatch in batchGenerator(validationBatchData, 1):
#        validate_accuracy, validate_confusion, validate_result = sess.run([accuracy, confusion, validate_result], feed_dict={
#        validate_result, a2,a3,a4,y,x2 = sess.run([result_argmax,result_argmax2,result_argmax3,result_argmax4,y_conv,x_check], feed_dict={
	validate_result = sess.run(result_argmax, feed_dict={
            x: miniBatch[0],
            keep_prob: 1.0})
        print("validate_result: " + str(validate_result+4))# + "\targmax_2: " + str(a2) + "\targmax_3: " + str(a3) + "\targmax_4: " + str(a4) + "\ty_conv: " +str(y))
#        for board in x2:
#            drawBoard(board)
        for i in validate_result:
            result[i] = result[i] + 1
#        final_confusion = final_confusion + validate_confusion
#        sumOfValidations = sumOfValidations + validate_accuracy
#        amountOfValidations = amountOfValidations + 1

    #print("validation: "  + str(sumOfValidations/amountOfValidations))
    #print("confusion matrix:\n" + str(final_confusion))
    print("result: " + str(result))
    sys.stdout.flush()


