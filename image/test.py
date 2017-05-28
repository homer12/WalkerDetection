# coding:utf-8

from time import sleep


fileName = 'www.txt'

for i in range(6):
	file = open(fileName,"a")
	file.write(str(i)+"\n")
	file.close()
	sleep(5)