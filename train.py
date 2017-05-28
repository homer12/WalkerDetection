#coding=utf-8

import cv2
import numpy as np
import feature
import os
import copy

trainingImgList = []        #训练中要用到的积分图列表
trainingLabelList = []      #训练中要用到的label列表



#先处理脸部训练图片
'''
print " Reading Face Pictures"
faceImgList = os.listdir('image/training_faces')
for file in faceImgList:
    fullImgPath = 'image/training_faces/' + file
    img = cv2.imread(fullImgPath)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    integral_img = feature.cal_integral_image(gray)
    
    trainingImgList.append(integral_img)
    trainingLabelList.append(1)
print " Done! "
'''


#再处理非脸部训练图片
'''
print " Reading Non Face Pictures "
nonFaceImgList = os.listdir('image/training_nonfaces')
for file in nonFaceImgList:
    fullImgPath = 'image/training_nonfaces/' + file
    img = cv2.imread(fullImgPath)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    integral_img = feature.cal_integral_image(gray)
    
    trainingImgList.append(integral_img)
    trainingLabelList.append(0)
print " Done! "
'''

#从文件rect.txt中读取rect_list
print " Reading rect_list "
rect_list = feature.read_rect_from_file('rect.txt')
print " Done! "


#训练样例，返回被选中的特征列表
print " Begin Training "
#def cascade_train( featureList, maxFalsePositivePerLayer, minDetectionRatePerLayer , targetFalsePositive)
#cas = feature.cascade_train(rect_list, 0.4, 0.9, 0.05)
cas = feature.cascade_train(rect_list, 0.9, 0.05, 0.9)
print " Done! "



#将被选中特征列表写入文件
feature.write_cascade_to_file(cas)








cv2.waitKey(0)
cv2.destroyAllWindows()  