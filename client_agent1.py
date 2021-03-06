import random
import numpy as np
import pickle
from socket import *
import sys
import time
import datetime

import select
#from termios import tcflush, TCIFLUSH
#sys.stdout = open('/files1d/1501858/Desktop/snake_output.txt', 'w')
#logFile = open('/files1d/1501858/Desktop/snakeNN/snake_client_output/snake_output_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + ".log", 'w')

#Server variables
serverName= '10.100.6.2' #'146.141.56.15' #'10.100.3.11'#'146.141.56.15' #'10.10.187.175'
serverPort=12000

# Setting up connection to server
clientSocket = socket(AF_INET, SOCK_STREAM)  # define and open client socket on client (with IPV4 and TCP)
clientSocket.connect((serverName, serverPort))  # connecting to server

#Boards Parameters
BOARD_WIDTH=50
BOARD_HEIGHT = 50

brushSuperApple = 7 #np.float32(6)
brushNormalApple = 6 #np.float32(5)
brushEnemyHead = 2 #np.float32(2)
brushEnemyBody = 1 #np.float32(1)
brushWinnerHead = 5 #np.float32(4)
brushWinnerBody = 4
brushBorder = 3

def addSnakeToMap(snakeEdges, gameMap, isMySnake):
	for i in range(len(snakeEdges)-1):
		if(snakeEdges[i][0] == snakeEdges[i+1][0]):
			if(snakeEdges[i][1] < snakeEdges[i+1][1]):
				for j in range(snakeEdges[i+1][1] - snakeEdges[i][1] + 1):
					gameMap[snakeEdges[i][0]][snakeEdges[i][1]+j] = ((not isMySnake)*brushEnemyBody) + (isMySnake*brushWinnerBody    )
			else:
				for j in range(snakeEdges[i][1] - snakeEdges[i+1][1] + 1):
					gameMap[snakeEdges[i][0]][snakeEdges[i+1][1]+j] = ((not isMySnake)*brushEnemyBody) + (isMySnake*brushWinnerBody    )
		else:
			if(snakeEdges[i][0] < snakeEdges[i+1][0]):
				for j in range(snakeEdges[i+1][0] - snakeEdges[i][0] + 1): 
					gameMap[snakeEdges[i][0]+j][snakeEdges[i][1]] = ((not isMySnake)*brushEnemyBody) + (isMySnake*brushWinnerBody    )
			else:
				for j in range(snakeEdges[i][0] - snakeEdges[i+1][0] + 1): 
					gameMap[snakeEdges[i+1][0]+j][snakeEdges[i][1]] = ((not isMySnake)*brushEnemyBody) + (isMySnake*brushWinnerBody    )
	return gameMap


def sphere (winningSnake, board):
	if winningSnake[0]== "invisible":
		isInvisible=True
	else:
		isInvisible = False

	# head = tuple(map(int, winningSnake[3 + 2*isInvisible].split(',')))
	head = winningSnake[3 + 2 * isInvisible]
	newHead = (head[1],BOARD_HEIGHT - 1 - head[0])

	if (newHead[1]<(BOARD_HEIGHT+1)/2):
		board = np.roll(board,(newHead[1]+2+(BOARD_HEIGHT+1)/2),axis=0)
	elif (newHead[1]>(BOARD_HEIGHT+1)/2):
		board = np.roll(board, (newHead[1]-(BOARD_HEIGHT+1)/2), axis=0)

	if (newHead[0]<(BOARD_WIDTH+1)/2):
		board = np.roll(board,((BOARD_WIDTH+1)/2-newHead[0]),axis=1)
	elif (newHead[0]>(BOARD_WIDTH+1)/2):
		board = np.roll(board, (BOARD_WIDTH-newHead[0]+2+(BOARD_WIDTH+1)/2), axis=1)

	# firstKink = tuple(map(int, winningSnake[7 + 2*isInvisible].split(',')))
	firstKink = winningSnake[4 + 2 * isInvisible]
	newfirstKink = (firstKink[1], BOARD_HEIGHT - 1 - firstKink[0])
	if (newHead[0] < newfirstKink[0]): #dir=left
		board = np.rot90(board,3)
		board = np.roll(board, -1, axis=1)
		board = np.roll(board, -3, axis=0)
	elif (newHead[0] > newfirstKink[0]): #dir=right
		board =  np.rot90(board, 1)
		board = np.roll(board, -2, axis=0)
	elif (newHead[1] < newfirstKink [1]): #dir=down
		board = np.rot90(board, 2)
		board = np.roll(board, -3, axis=0)
	else:                               #dir=up
		board = np.roll(board, -1, axis=1)
		board = np.roll(board, -2, axis=0)

	return board

def drawBoard(board, boardSize):
	for i in range(boardSize):
		if board[i] == 0:
			print(" "),
		else:
			print(board[i]),
 
		if (i%52==51):
			print("")
 
def drawUnflattendBoard(board):
	for row in board:
		for block in row:
			if block == 0:
				print(" "),
			else:
				print(block)
		print("")

def flushInput():
	line = ""
	while(select.select([sys.stdin,],[],[],0.0)[0]):# or line != "-1 -1"):
		line = raw_input()
		#logFile.write("first while loop... " + str(time.time()) + "\t" + line + "\n")

#	while(line != "-1 -1"):
#		line = raw_input()
#		logFile.write("second while loop... " + str(time.time())  + "\t" + line + "\n")
	
#	return line

if __name__ == "__main__":
#def playGame():
	#print("asdasd")
	#sys.stdout.flush()
	line = raw_input()
	split = line.split(" ")
	numSnakes = int(split[0])
	
	# Setting up connection to server
#	clientSocket = socket(AF_INET, SOCK_STREAM)  # define and open client socket on client (with IPV4 and TCP)
#	clientSocket.connect((serverName, serverPort))  # connecting to server

	#sleep(10)
#	if(not sys.stdin.isatty()):
#		logFile.write("flushing stdin...")
#		tcflush(sys.stdin, TCIFLUSH)
#		logFile.write("stdin flushed\n")
#	else:
#		logFile.write("stdin was empty\n")

#	while(select.select([sys.stdin,],[],[],0.0)[0]):# or line != "-1 -1"):
#		line = raw_input()
#		logFile.write("first while loop... " + str(time.time()) + "\t" + line + "\n")

#	while(line != "-1 -1"):
#		line = raw_input()
#		logFile.write("second while loop... " + str(time.time())  + "\t" + line + "\n")

	flushInput()
	
#	firstRound = True
	
	while(True):
		gameMap = np.zeros((50, 50), dtype=np.int8)
						  # dtype=np.float32)  # [[0 for i in xrange(50)] for j in xrange(50)] #gameMap is a 50x50 zeroes
		allSnakes = []

#		if (line != "-1 -1"):
		line = raw_input()

		if "Game Over" in line:
			break
		
		superApple = tuple(map(int, line.split(' ')))
		startTime = time.time()
		if ((superApple[0] != -1) and (superApple[1] != -1)):
			gameMap[superApple[0]][superApple[1]] = brushSuperApple
		
		normalApple = tuple(map(int, raw_input().split(' ')))
		gameMap[normalApple[0]][normalApple[1]] = brushNormalApple
		# gameMap[mySnake[3+(mySnake[0]=="invisible")][0]][mySnake[3+(mySnake[0]=="invisible")][1]] = mySnakeHeadBrush
		
		mySnakeIndex = int(raw_input())
		
		enemySnake = []
		for i in range(numSnakes):
			line = raw_input()
			if(i == mySnakeIndex):
				mySnake=line.split(' ')
				# mySnakeCoords = []
				#print("mysnake before for loop" + str(mySnake))
				for count,coords in enumerate(mySnake[3+(mySnake[0]=="invisible"):]):
					mySnake[3 + (mySnake[0] == "invisible")+count] = tuple(map(int, coords.split(',')))
					# mySnakeCoords.append(tuple(map(int, coords.split(','))))
				#print("mysnake after for loop" + str(mySnake))
				#print("mySnake[4+(mySnake[0]=='invisible'):] " + str(mySnake[3+(mySnake[0]=="invisible"):]))
				if(mySnake[0] != "dead"):
					#addSnakeToMap(mySnakeCoords)
					#print("1111111111111111111111111")
					gameMap = addSnakeToMap(mySnake[3+(mySnake[0]=="invisible"):], gameMap, True)
					#print("22222222222222222222222222")
					gameMap[mySnake[3+(mySnake[0]=="invisible")][0]][mySnake[3+(mySnake[0]=="invisible")][1]] = brushWinnerHead #overwriting our head to world map
				#drawUnflattendBoard(gameMap)
				#print (gameMap)
			
			else:
				enemySnake.append(line.split(' '))
				#enemySnakeCoords = []
				for count,coords in enumerate(enemySnake[-1][3+(enemySnake[-1][0]=="invisible"):]):
					enemySnake[-1][3 + (enemySnake[-1][0] == "invisible")+count] = tuple(map(int, coords.split(',')))
					#enemySnakeCoords.append(tuple(map(int, coords.split(','))))
				if(enemySnake[-1][0] != "dead"):
					#addSnakeToMap(enemySnakeCoords)
					#print("333333333333333333333333333333333333333333")
					#print("enemy snake: " + str(enemySnake))
					#print("enemySnake[-1][3+(enemySnake[-1][0]=='invisible'):] " + str(enemySnake[-1][3+(enemySnake[-1][0]=="invisible"):]))
					gameMap = addSnakeToMap(enemySnake[-1][3+(enemySnake[-1][0]=="invisible"):],gameMap, False)
					#print("4444444444444444444444444444444444444444444")
					gameMap[enemySnake[-1][3 + (enemySnake[-1][0] == "invisible")][0]][enemySnake[-1][3 + (enemySnake[-1][0] == "invisible")][1]] = brushEnemyHead


		#pad here
		gameMap = np.pad(gameMap, 1, 'constant', constant_values=(brushBorder))
		#sphere here
		spheredGameMap = sphere(mySnake, gameMap)
		spheredGameMap = spheredGameMap.flatten()
		# gameMapToSend = pickle.dump(spheredGameMap)
		# print(spheredGameMap.shape)
		pickledSpheredGameMap = pickle.dumps(spheredGameMap)
		clientSocket.send(str(sys.getsizeof(pickledSpheredGameMap)))
		clientSocket.send(pickledSpheredGameMap)

		# print("Waiting for move from server...")
		replyAnswerSerialized = clientSocket.recv(64).decode('utf-8')
		# print("Received for move from server")
		#replyAnswer = pickle.loads(replyAnswerSerialized)
		#print(replyAnswer)

		#if (firstRound):
		#	firstRound = False
		#	logFile.write("skipped first round\n")
		#	continue
		
		sys.stdout.flush()
		if(replyAnswerSerialized == "4"):
			print("6")
			#logFile.write(str(replyAnswerSerialized)+"\t")
		elif(replyAnswerSerialized == "6"):
			print("4")
			#logFile.write(str(replyAnswerSerialized)+"\t")
		elif(replyAnswerSerialized == "5"):
			print("5")
			#logFile.write(str(replyAnswerSerialized)+"\t")
		else:
			print("0")
			#logFile.write(str(replyAnswerSerialized)+"------------------------\t")

#		sys.stdout.flush()
#		print("4")
#		logFile.write("4\t")
#		sys.stdout.flush()

#		print(replyAnswerSerialized)
		#logFile.write(str(replyAnswerSerialized)+"\t")
		#logFile.write("time per step: " + str(time.time()-startTime) + "\n")
		#		print("time per step: " + str(time.time()-startTime))
		
	# print ("Closing Connection...")
	clientSocket.close()
	# print("Connection closed")



     #    return spheredGameMap
	# return

