import random
import numpy as np
import pickle
from socket import *
import sys

#Server variables
serverName='146.141.56.15' #'10.10.187.175'
serverPort=12000

# Setting up connection to server
clientSocket = socket(AF_INET, SOCK_STREAM)  # define and open client socket on client (with IPV4 and TCP)
clientSocket.connect((serverName, serverPort))  # connecting to server


if __name__ == "__main__":

	print("importing validation data file...")
	fileLocation = "/mnt/snake/snakeNN/snakeNN_data5/"
	innerFolder=""
	validationBatchData = [np.load(fileLocation + "validationData/"+ innerFolder + 'validationDataBoards' + str(1) + ".npy"), np.load(fileLocation + "validationData/" + innerFolder + 'validationDataMoves' + str(1) + ".npy")]
	
	for x,y in zip(validationBatchData[0], validationBatchData[1]):
		pickledX = pickle.dumps(x)
	        clientSocket.send(str(sys.getsizeof(pickledX)))
        	clientSocket.send(pickledX)

        	# print("Waiting for move from server...")
	        replyAnswerSerialized = clientSocket.recv(32)

		print("reply from server: " + str(replyAnswerSerialized) + "\treal answer: " + str(y))

	clientSocket.close()
