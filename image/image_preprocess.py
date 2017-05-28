# -*- coding:utf-8 -*-

import numpy as np
import cv2
import os

#输入文件夹的名字
folderName = 'nonfaces'
imgList =  os.listdir(folderName)

print len(imgList)

for i in range(len(imgList)):
	fullReadPath = folderName + '/' +imgList[i]
	if( cmp( fullReadPath[-3:] , "png" ) != 0 and cmp( fullReadPath[-3:] , "PNG") != 0 ):
		continue
	
	#读取、调整规格、灰度化
	image = cv2.imread(fullReadPath)
	res = cv2.resize( image, (20, 34))
	gray = cv2.cvtColor(res , cv2.COLOR_BGR2GRAY)
	
	#输出文件的名字
	fullWritePath = 'training_' +  folderName + '/' + imgList[i]
	cv2.imwrite( fullWritePath, gray)
	
	#控制张数
	#if( i >= 400 ):	break			