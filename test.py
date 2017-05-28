# -*- coding:utf-8 -*-
import feature
import numpy as np
import cv2
import os
import math
import feature
from numpy import select

def isFace(selectedFeatures, img):
    #beta = error / ( 1-error )
    #alpha = log(1/beta)
    
    left = 0.0
    right = 0.0
    
    for f in selectedFeatures:
        beta = f.error  / ( 1.0-f.error )
        alpha = math.log( 1.0/beta )        
        
        #def h(image, feature, p, thres )
        left += alpha * feature.h(img,f.feature,f.p,f.thres)
        right += 0.5 * alpha
        
    if( left >= right ):    return 1
    else:                   return 0
    
    return


#从文件中读取Cascade
cas = feature.read_cascade_from_file()



#读取所有图片
testingImgList = []
testingLabelList = []
 
print 'Reading face pictures'
testingImgFolder = 'image/testing_faces'
faceImgList = os.listdir(testingImgFolder)
for file in faceImgList:
    fullImgPath = testingImgFolder + '/' + file
    img = cv2.imread(fullImgPath)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    integral_img = feature.cal_integral_image(gray)
    
    testingImgList.append(integral_img)
    testingLabelList.append(1)
print 'Done'


print 'Reading nonface pictures'
testingImgFolder = 'image/testing_nonfaces'
nonFaceImgList = os.listdir(testingImgFolder)
for file in nonFaceImgList:    
    fullImgPath = testingImgFolder + '/' + file
    img = cv2.imread(fullImgPath)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    integral_img = feature.cal_integral_image(gray)
    
    testingImgList.append(integral_img)
    testingLabelList.append(0)
print 'Done'



#统计falsePositive和detectiongRate
truePositive = 0.0
trueNegative = 0.0
falsePositive = 0.0
falseNegative = 0.0

for i in range(len(testingImgList)):
    img = testingImgList[i]
    label = testingLabelList[i]
    cx = feature.cascade_classify(cas, img)
    
    if( label == 1 and cx == 1 ):   truePositive += 1
    elif ( label == 1 and cx == 0 ):    falseNegative += 1
    elif ( label == 0 and cx == 1): falsePositive += 1
    elif ( label == 0 and cx == 0): trueNegative += 1

print 'Sum =',len(testingImgList)
print 'Positive =',truePositive+falseNegative
print 'Negative =',trueNegative+falsePositive
print 'TruePositive =',truePositive
print 'TrueNegative =',trueNegative
print 'FalsePositive =',falsePositive
print 'FalseNegative =',falseNegative
print 'Correct Detection Rate =' , truePositive / (truePositive+falseNegative)
print 'False Positive Rate =', falsePositive / (falsePositive+trueNegative)