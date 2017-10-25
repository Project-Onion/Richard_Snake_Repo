import numpy as np
import time

# boards=numpy.load('./snake_json_files/trainingDataBoards.npy')

# def drawBoard(boards, move, num):#indexArray, num):
# 	row = 0
# 	head=(-1,-1)
# 	for i in boards[num]:
# 		row = row + 1
# 		column = 0
# 		for j in i:
# 			column = column + 1
# 			if j == 0:
# 				print(" "),
# 			else:
# 				print(j),
# 				if j==4:
# 					head=(column,50 - row)
# 		print("")
# 	print("-------------------------------------------------------------")
# 	print("head: " + str(head))
# 	print ("move: "+ str(move[num]))
# 	#print ("index: " + str(indexArray[num]))

def drawBoard(boards, move, num):#indexArray, num):
	print("")
	row = 1
	column = 0
	head=(-1,-1)
	for i in range(2704):
		column = column + 1

		if boards[num][i] == 0:
			print(" "),
		else:
			print(boards[num][i]),
			if boards[num][i]==4:
				head=(row,column)

		if (i%52==51):
			row = row + 1
			column = 0
			print("")

	print("")
	print("-------------------------------------------------------------")
	print("head: " + str(head))
	print ("move: "+ str(move[num]))
	#print ("index: " + str(indexArray[num]))



def movieBoard(allBoards, move): #, indexArray):
	print ("starting movie")
	for i in range(0,len(allBoards)):
		drawBoard(allBoards, move, i)#indexArray, i)
		time.sleep(0.5)
