# -*-coding:utf-8-*-

import cv2
import numpy as np
import os
import math
import copy


def debug():
    while(1):
        c = raw_input("Input g to continue ...\n")
        if c == 'g' : return


#类的说明
'''
Rectangle = x0(num) , y0(num) , x1(num) , y1(num) , rType(num)
SeFeature = error(num) , feature(Rectangle) , p(num) , thres(num)
Layer = numberOfFeatures(num) , seFeatureList(list of SeFeature) , thres(num)
Cascade = numberOfLayers(num) , layerList(list of Layer)
'''


class Rectangle:    
    def __init__(self, x0=-1, y0=-1, x1=-1, y1=-1, rType=-1):
        self.x0 = x0
        self.y0 = y0    #(x0,y0) = coordinate of the left-up vertex
        self.x1 = x1
        self.y1 = y1    #(x1,y1) = coordinate of the right-down vertex
        self.rType = rType
        '''
        type 0 represents 1*2
        type 1 represents 2*1
        type 2 represents 1*3
        type 3 represents 3*1
        type 4 represents 2*2
        '''
    
    def printPoints(self):
        print '(' + str(self.x0) + ',' + str(self.y0) + ',' + str(self.x1) + ',' + str(self.y1) + ')'
        
    def toString(self):
        return str(self.x0) + '\t' + str(self.y0) + '\t' + str(self.x1) + '\t' + str(self.y1) + '\t' + str(self.rType) + '\n'

#每一个被选中的feature，包含了位置，误差权重，不等式方向，分类阈值
class SeFeature:
    def __init__(self, error=0.0, feature=Rectangle(), p=0.0, thres=0.0 ):
        self.error = error
        self.feature = feature
        self.p = p
        self.thres = thres
        
    def toString(self):
        return str(self.error) + '\t' +  str(self.feature.x0) + '\t' + str(self.feature.y0) + \
        '\t' + str(self.feature.x1) + '\t' + str(self.feature.y1) + '\t' + str(self.feature.rType) + \
        '\t' + str(self.p) + '\t' + str(self.thres) + '\n'
    
    
#瀑布模型中的每一层分类器，包含了一个SeFeature列表，和一个分类阈值
class Layer:
    def __init__(self, numberOfFeatures=0, seFeatureList=[], thres=0.0):
        self.numberOfFeatures = numberOfFeatures
        self.seFeatureList = seFeatureList
        self.thres = thres
        
    def toString(self):
        resultStr = ""
        resultStr += str(self.numberOfFeatures) + '\n'
        
        for i in range(self.numberOfFeatures):
            curFeature = self.seFeatureList[i]
            #resultStr += curFeature.toString()
            resultStr += str(curFeature.error) + '\t' + str(curFeature.feature.x0) + '\t' + str(curFeature.feature.y0) + \
                '\t' + str(curFeature.feature.x1) + '\t' + str(curFeature.feature.y1) + '\t' + str(curFeature.feature.rType) + \
                '\t' + str(curFeature.p) + '\t' + str(curFeature.thres) + '\n'
        
        resultStr += str(self.thres) + '\n'
        
        return resultStr
        
        
class Cascade:
    def __init__(self, numberOfLayers=0, layerList=[]):
        self.numberOfLayers = numberOfLayers
        self.layerList = layerList
    
    def toString(self):
        resultStr = ""
        resultStr += str(self.numberOfLayers) + '\n'
        
        for i in range(self.numberOfLayers):
            curLayer = self.layerList[i]
            resultStr += curLayer.toString()
        
        return resultStr
        

    
def cal_number_of_features( w,h, s, t, rect_list, type ):
    #根据给定的边长，特征尺寸s和t，以及特征类型type，计算出各种矩形特征的顶点，然后填充到rect_list里
    for x0 in range(w):
        for y0 in range(h):        
            p = 1
            
            while 1:
                x1 = x0 + p * s - 1
                if( x1 >= w ):
                    break
                
                q = 1
                
                while 1:
                    y1 = y0 + q * t - 1
                    if( y1 >= h ):
                        break
                    
                    rec = Rectangle( x0, y0, x1, y1, type )
                    rect_list.append(rec)
                    
                    q = q+1
                
                p = p+1
            
def cal_vertices_of_features( w=24 , h=24 ):
    #计算在width*width子窗口中，所有五种矩形特征的位置
    rect_list = []
    cal_number_of_features(w,h, 1, 2, rect_list, 0)
    cal_number_of_features(w,h, 2, 1, rect_list, 1)
    cal_number_of_features(w,h, 1, 3, rect_list, 2)
    cal_number_of_features(w,h, 3, 1, rect_list, 3)
    cal_number_of_features(w,h, 2, 2, rect_list, 4)
    return rect_list

            

def write_rect_to_file(rect_list):
    #将计算好的所有特征位置输出到文件rect.txt中
    file = open("rect.txt" , 'w' )
    for rect in rect_list:
        file.write(rect.toString())
    file.close()
    

    
def read_rect_from_file(filename):
    #从指定的文件读入所有特征位置，构成特征位置列表
    file = open(filename , 'r')
    rect_list = []
    
    for line in file.readlines():
        lineList = line.strip().split('\t')
        rect = Rectangle( int(lineList[0]) , int(lineList[1]), int(lineList[2]), int(lineList[3]), int(lineList[4]) )
        rect_list.append(rect)
    
    file.close()
    return rect_list

def cal_integral_image(gray):
    #根据给定的一张灰度图，计算积分图
    h,w = gray.shape[:2]
    integral = np.zeros( [h,w] , dtype=int )
    s = np.zeros( [h,] , dtype=int )
    
    for x in range(w):
        for y in range(h):
            if( y == 0 ):   s[y] = gray[y][x]
            else:           s[y] = s[y-1] + gray[y][x]
            
            if( x == 0 ):   integral[y][x] = s[y]
            else:           integral[y][x] = integral[y][x-1] + s[y]
            
    return integral


#若给定的坐标是合法的，就返回对应的积分图数值。简化主函数中的边界检查语句
def integral_of_a_point( integral, x, y ):
    if( x>=0 and y>=0 ):
        return integral[y][x]
    else:
        return 0


#计算这两个顶点确定的矩形特征数值
def cal_a_rectangle( integral, x0, y0, x1, y1 ):
    A = integral_of_a_point(integral, x1, y1)
    B = integral_of_a_point(integral, x0-1, y0-1)
    C = integral_of_a_point(integral, x0-1, y1)
    D = integral_of_a_point(integral, x1, y0-1)
    return A-C-D+B


def test1( integral, gray):
    '''To test the correctness of the cal_integral_image function'''
    test_x, test_y = input("Input x and y : ")
    res = 0
    for x in range(test_x+1):
        for y in range(test_y+1):
            res += gray[y][x]
    print 'res =',res,'integral =',integral[y][x]
    

def test2( gray, integral, x0, y0, x1, y1 ):
    '''To test the correctness of the cal_a_rectangle function'''
    res = 0
    for x in range(x0,x1+1):
        for y in range(y0,y1+1):
            res += gray[y][x]
    print 'sum of integral = ',res
    print 'result of calculation = ',cal_a_rectangle(integral, x0, y0, x1, y1)

#根据不同的矩形特征，计算具体的特征数值
def cal_a_feature( integral, x0, y0, x1, y1, rType ):    
    if rType == 0:     # 1*2 (h*w)
        w = x1 - x0 + 1     #width          
        x2 = x0 + w/2
        
        # res = right - left
        A = integral_of_a_point(integral, x2-1, y1)
        B = integral_of_a_point(integral, x0-1, y1)
        C = integral_of_a_point(integral, x0-1, y0-1)
        D = integral_of_a_point(integral, x2-1, y0-1)
        E = integral_of_a_point(integral, x1, y0-1)
        F = integral_of_a_point(integral, x1, y1)        
        return 2*A-B+C-2*D+E-F
    
    elif rType == 1:     # 2*1 (h*w)
        h = y1 - y0 + 1        
        y2 = y0 + h/2
        
        #res = down - up
        A = integral_of_a_point(integral, x1, y2-1)
        B = integral_of_a_point(integral, x1, y0-1)
        C = integral_of_a_point(integral, x0-1, y0-1)
        D = integral_of_a_point(integral, x0-1, y2-1)
        E = integral_of_a_point(integral, x0-1, y1)
        F = integral_of_a_point(integral, x1, y1)
        return 2*A-B+C-2*D+E-F
        
    elif rType == 2:     #1*3 (h*w)
        w = x1 - x0 + 1
        x2 = x0 + w/3
        x3 = x2 + w/3
        
        #res = left + right - middle
        A = integral_of_a_point(integral, x3-1, y1)
        B = integral_of_a_point(integral, x2-1, y1)
        C = integral_of_a_point(integral, x0-1, y1)
        D = integral_of_a_point(integral, x0-1, y0-1)
        E = integral_of_a_point(integral, x2-1, y0-1)
        F = integral_of_a_point(integral, x3-1, y0-1)
        G = integral_of_a_point(integral, x1, y0-1)
        H = integral_of_a_point(integral, x1, y1)
        return -2*A + 2*B - C + D - 2*E + 2*F - G + H
     
    elif rType == 3:     #3*1 (h*w)
        h = y1 - y0 + 1
        y2 = y0 + h/3
        y3 = y2 + h/3
        
        #res = up + down - middle
        A = integral_of_a_point(integral, x1, y3-1)
        B = integral_of_a_point(integral, x1, y2-1)
        C = integral_of_a_point(integral, x1, y0-1)
        D = integral_of_a_point(integral, x0-1, y0-1)
        E = integral_of_a_point(integral, x0-1, y2-1)
        F = integral_of_a_point(integral, x0-1, y3-1)
        G = integral_of_a_point(integral, x0-1, y1)
        H = integral_of_a_point(integral, x1, y1)
        return -2*A + 2*B - C + D - 2*E + 2*F - G + H
    
    elif rType == 4:     # 2*2 (h*w)
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        
        x2 = x0 + w/2
        y2 = y0 + h/2
        
        #res = (-45degree) - (45degree)
        A = integral_of_a_point(integral, x2-1, y2-1)
        B = integral_of_a_point(integral, x1, y2-1)
        C = integral_of_a_point(integral, x1, y0-1)
        D = integral_of_a_point(integral, x2-1, y0-1)
        E = integral_of_a_point(integral, x0-1, y0-1)
        F = integral_of_a_point(integral, x0-1, y2-1)
        G = integral_of_a_point(integral, x0-1, y1)
        H = integral_of_a_point(integral, x2-1, y1)
        I = integral_of_a_point(integral, x1, y1)
        return 4*A-2*B+C-2*D+E-2*F+G-2*H+I
        
        
        
        
    
def train( imageList, labelList, featureList , T):     # imageList = list of integral images        
    #T = 5     #修改后，T由用户指定
    selectedFeatures = []       #选出的特征列表，列表中每一个元素都是Rectangle类型 
    
    #初始化权重
    nPositive = 0
    nNegative = 0
    weight = [0.0] * len(imageList)
    
    for label in labelList:
        if label == 1:  nPositive += 1
        else        :   nNegative += 1
    
    for i in range(len(labelList)):
        if labelList[i] == 1:
            weight[i] = 1.0 / (2.0*nPositive)
        else:
            weight[i] = 1.0 / (2.0*nNegative)
            
    
    #For each round, a feature will be selected
    #And be removed from the original feature list
    
    
    for t in range(T):
        #每一轮计算所有特征可以达到的最好误差
        #选择最好误差最小的那个作为这一轮的“被选择特征”
        #并更新样例的权重
        #print " Selecting the Feature " + str(t+1)
        
        bestError = 99999.0
        bestFeature = Rectangle()
        bestP = 0           #方向，p*特征值 < p*阈值 = 判定为正
        bestThres = 0       #某一个特定特征的判定阈值
        
        #对上一轮经过调整过的样例权重，进行正则化
        #并用正则化过的样例权重计算 总正样例权重和T_plus 和 总负样例权重和T_minus
        T_plus = 0.0
        T_minus = 0.0
        sumWeight = 0.0
        for w in weight:
            sumWeight += w
        for i in range(len(weight)):
            weight[i] = weight[i] / sumWeight
            if( labelList[i] == 1):
                T_plus += weight[i]
            else:
                T_minus += weight[i]
                
        #显示所有图片的权值
        #print weight
        
        #存放对于最佳特征，每个照片的计算值
        picValueList = [0.0] * len(imageList)
        
        #xx = 0
        for feature in featureList:                  
            #print xx
            #xx += 1
            
            #featureValueOnEachExample这个列表存储的是对于某个特征，所有样例计算出来的特征值
            featureValueOnEachExample = []
            
            for image in imageList:
                value = cal_a_feature(image, feature.x0, feature.y0, feature.x1, feature.y1, feature.rType)
                featureValueOnEachExample.append(value)
            
            #对featureValueOnEachExample列表进行argsort，结果保存到sortedIndices中
            #sortedIndices中存储的是featureValueOnEachExample列表的序号，按照对应特征值从低到高的顺序
            sortedIndices = np.array(featureValueOnEachExample).argsort()
            
            #从第一个元素的特征值-0.1开始分，这里就是把所有样本都判定为同正或同负            
            thres = featureValueOnEachExample[sortedIndices[0]]-0.1
            S_plus = 0.0
            S_minus = 0.0
            
            #print 'T_minus=',T_minus
            #print 'T_plus=',T_plus
            
            A = S_plus + ( T_minus - S_minus )  #大于该阈值判定为正的误判率 = 小于且为正的权重 + 大于且为负的权重
            B = S_minus + ( T_plus - S_plus)    #大于该阈值判定为负的误判率 = 小于且为负的权重 + 大于且为正的权重
            
            if A < B :      #若A小，则大于判正
                currentError = A        
                currentP = -1
                currentThres = thres
            else:
                currentError = B
                currentP = 1
                currentThres = thres
                
                
                
            #i start from 0 to len(imageList)-2
            #because there are len(imageList)-1 intervals
            #for each i, the threshold = an intermediate between sortedIndices[i] and sortedIndices[i+1] 
            for i in range(len(imageList)-1):
                thres = featureValueOnEachExample[sortedIndices[i]]+0.1
                
                if( labelList[sortedIndices[i]] == 1 ):
                    S_plus += weight[sortedIndices[i]]
                else:
                    S_minus += weight[sortedIndices[i]]
                    
                A = S_plus + ( T_minus - S_minus )
                B = S_minus + ( T_plus - S_plus)
                
                if A < currentError:
                    currentError = A
                    currentP = -1
                    currentThres = thres
                
                if B < currentError:
                    currentError = B
                    currentP = 1
                    currentThres = thres
            
            #After we get the best threshold
            #if the best error rate of current feature is less than that of all the last features
            #then update the bestError and bestFeature and the direction of inequality and threshold
            
            
            
            if( currentError < bestError ):
                bestError = currentError
                bestFeature = feature
                bestP = currentP
                bestThres = currentThres
                picValueList = featureValueOnEachExample[:]                
        
        
        #输出图片特征值数组
        #print picValueList
            
        
        #根据被选出来的feature，构建一个SeFeature类型的数据selectedFeature
        bestError += 0.0000000001         #为了防止error为0出现除0错误  
        selectedFeature = SeFeature()
        selectedFeature.error = bestError
        selectedFeature.feature = bestFeature
        selectedFeature.p = bestP
        selectedFeature.thres = bestThres
        
        #输出分类器信息
        #print '阈值:',selectedFeature.thres,'误差:',selectedFeature.error,'符号:',selectedFeature.p
        
        #Add to list
        selectedFeatures.append(selectedFeature)
        
        #Removed from original feature list
        featureList.remove(bestFeature)
        
        #Update the weights
        #print '判断错误的样例编号'
        for i in range(len(imageList)):
            if( h( imageList[i], bestFeature, bestP, bestThres ) == labelList[i] ):
                weight[i] = weight[i] * ( bestError / (1.0-bestError))
            #else:
                #输出判断错误的样例
                #print i
                
        
        #输出调整后的权重数组（未归一化）
        #print weight
        
        print " Done Selecting the Feature " + str(t+1)
            
    return selectedFeatures    



def h(image, feature, p, thres ):
    fx = cal_a_feature(image, feature.x0, feature.y0, feature.x1, feature.y1, feature.rType)
    if( p*fx < p*thres ):
        return 1
    else:
        return 0
 

def write_selectedfeatures_to_file( selectedFeatures ):    
    #将计算好的selectedFeatures输出到SeFeature.txt
    file = open( 'SeFeature.txt', 'w' )
    for feature in selectedFeatures:
        file.write(feature.toString())
    file.close()


def read_selectedfeatures_to_file( filename ):
    #从指定文件读入所有selected features
    file = open( filename , 'r' )
    selectedFeatures = []

    for line in file.readlines():
        lineList = line.strip().split('\t')
        feature = SeFeature( float(lineList[0]) , \
                    Rectangle( int(lineList[1]), int(lineList[2]), int(lineList[3]), int(lineList[4]), int(lineList[5]) ), \
                    int(lineList[6]), float(lineList[7]) )
        selectedFeatures.append(feature)
    
    return selectedFeatures
        

def cascade_train( featureList, maxFalsePositivePerLayer, minDetectionRatePerLayer , targetFalsePositive):
    #读入照片
    print 'Reading Pictures'
    
    #初始化正例集合
    positiveImgList = []
    faceImgFileList = os.listdir('image/training_faces')
    for picFile in faceImgFileList:        
        fullImgPath = 'image/training_faces/' + picFile
        img = cv2.imread(fullImgPath)
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        integral_img = cal_integral_image(gray)
        positiveImgList.append(integral_img)    
    positiveLabelList = [1] * len(positiveImgList)
     
    #初始化负例集合
    negativeImgList = []
    nonfaceImgFileList = os.listdir('image/training_nonfaces')
    for picFile in nonfaceImgFileList:        
        fullImgPath = 'image/training_nonfaces/' + picFile
        img = cv2.imread(fullImgPath)
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        integral_img = cal_integral_image(gray)
        negativeImgList.append(integral_img)
    negativeLabelList = [0] * len(negativeImgList)
    
    #整合正负集合
    trainingImgList = positiveImgList[:]
    trainingImgList.extend(negativeImgList)
    trainingLabelList = positiveLabelList[:]
    trainingLabelList.extend(negativeLabelList)
    
    
    
    #初始化当前假正率和当前检测率
    curFalsePositive = 1.0
    curDetectionRate = 1.0
    
    #初始化空的瀑布类
    cas = Cascade()

    
    
    #进入瀑布分类器的训练，退出条件要么训练出符合目标(假正率低于目标)的分类器
    #要么无法在有限的时间代价下训练出需要的分类器
    cLayer = 0
    while curFalsePositive > targetFalsePositive :
        print 'curFP =',curFalsePositive,'curDR =',curDetectionRate
        print 'proceed training Layer'
        
        
        T = 0
        maxLimitT = 15      #限制的每层最大分类器数量
        layer = Layer()     #初始化一个新layer
        lastFalsePositive = curFalsePositive
        lastDetectionRate = curDetectionRate
        
        #进入每一个Layer的训练        
        print 'Training Layer',cLayer
        cLayer+=1
        while curFalsePositive > lastFalsePositive * maxFalsePositivePerLayer :
            T = T+1
            print 'T =',T
            print 'T =',T
            print 'T =',T
            
            #如果某一层分类器数量超过上限还没有得到理想的结果
            #就终止训练过程
            if( T > maxLimitT ):
                print 'Exceed the Limit of layers'
                return cas
            
            #用P和N训练一个含有T个分类器的Layer
            print 'Use P and N to train'
            layer.seFeatureList = train(trainingImgList, trainingLabelList, featureList, T)
            layer.numberOfFeatures = len(layer.seFeatureList)
            layer.thres = 0.0
            for f in layer.seFeatureList:
                err = f.error
                beta = err / (1.0-err)
                alpha = math.log(1.0/beta)
                layer.thres += 0.5*alpha
            
            #在检验集合(validation set)上评估这个Layer的FP和DR
            print 'Validating'
            (curFalsePositive,curDetectionRate) = evaluate_from_validation(layer)
            print 'curFP =',curFalsePositive,'curDR =',curDetectionRate
            
            #如果分类器无法达到误检率要求
            #while( curFalsePositive > lastFalsePositive * maxFalsePositivePerLayer ):
                #升高layer的阈值
                #print 'Raising the threshold of the layer'
                #layer.thres += 0.5
                #(curFalsePositive,curDetectionRate) = evaluate_from_validation(layer)
                #print 'curFP =',curFalsePositive,'curDR =',curDetectionRate

            #如果这个分类器无法达到检测率要求
            while( curDetectionRate < lastDetectionRate * minDetectionRatePerLayer ):
                #就降低layer的阈值
                print 'Loweing the threshold of the layer'
                layer.thres -= 0.5
                (curFalsePositive,curDetectionRate) = evaluate_from_validation(layer)
                print 'curFP =',curFalsePositive,'curDR =',curDetectionRate
            
            
        #这个layer训练完毕，添加到Cascade中
        cas.numberOfLayers += 1
        cas.layerList.append(layer)
        
        #将N清空，选取当前Cascade无法正确分类的non-face图片组成N
        #这里为了优化运算速度，只用最新的layer用来做non-face图片筛选器
        lastNegativeImgLast = negativeImgList[:]
        negativeImgList = []
        negativeLabelList = []
        for img in lastNegativeImgLast:
            classifyResult = layer_classify(layer, img)
            
            #分类错误
            if( classifyResult == 1 ):
                negativeImgList.append(img)
                negativeLabelList.append(0)
        #正负整合
        trainingImgList = positiveImgList[:]
        trainingImgList.extend(negativeImgList)
        trainingLabelList = positiveLabelList[:]
        trainingLabelList.extend(negativeLabelList)
        
        print cLayer-1,'layer over'
        print 'curFP =',curFalsePositive
        print 'curDr =',curDetectionRate
    
    #cascade训练完毕，返回
    return cas
            
            
            
            
def evaluate_from_validation(layer):
    #从validation set中读取图片
    #初始化正例集合
    positiveImgList = []
    faceImgFileList = os.listdir('image/vali_faces')
    for picFile in faceImgFileList:        
        fullImgPath = 'image/vali_faces/' + picFile
        img = cv2.imread(fullImgPath)
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        integral_img = cal_integral_image(gray)
        positiveImgList.append(integral_img)    
    positiveLabelList = [1] * len(positiveImgList)
    
    #初始化负例集合
    negativeImgList = []
    nonfaceImgFileList = os.listdir('image/vali_nonfaces')
    for picFile in nonfaceImgFileList:        
        fullImgPath = 'image/vali_nonfaces/' + picFile
        img = cv2.imread(fullImgPath)
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        integral_img = cal_integral_image(gray)
        negativeImgList.append(integral_img)
    negativeLabelList = [0] * len(negativeImgList)
    
    
    #整合正例和负例
    testingImgList = positiveImgList[:]
    testingImgList.extend(negativeImgList)
    
    testingLabelList = positiveLabelList[:]
    testingLabelList.extend(negativeLabelList)
    
    
    #统计FP和DR
    truePositive = 0.0
    trueNegative = 0.0
    falsePositive = 0.0
    falseNegative = 0.0
    
    
    #遍历validation set
    for i in range(len(testingImgList)):
        #Read img and label
        img = testingImgList[i]
        label = testingLabelList[i]
        
        result = layer_classify(layer, img)
        
        if( label == 1 and result == 1 ):   truePositive += 1.0
        elif( label == 1 and result == 0 ): falseNegative += 1.0
        elif( label == 0 and result == 1 ): falsePositive += 1.0
        elif( label == 0 and result == 0 ): trueNegative += 1.0
        
        
    return  falsePositive/(falsePositive+trueNegative) , truePositive/(truePositive+falseNegative)
    

#用一个Layer去分类一个img
def layer_classify(layer,img):        
    totalSum = 0.0
    for f in layer.seFeatureList:
        beta = f.error / (1.0-f.error)
        alpha = math.log( 1.0/beta )
        #h(image, feature, p, thres )
        totalSum += alpha * h(img,f.feature,f.p,f.thres)
    
    if( totalSum < layer.thres ):
        #总和小于layer的阈值，即不被这个layer判正，返回0
        return 0    
    else:
        return 1
    

#用一个Cascade去分类一个img
def cascade_classify( cas,img ):
    for layer in cas.layerList:
        totalSum = 0.0
        for f in layer.seFeatureList:
            beta = f.error / (1.0-f.error)
            alpha = math.log( 1.0/beta )
            totalSum += alpha * h(img,f.feature,f.p,f.thres)
        if( totalSum < layer.thres ):
            #总和小于layer的阈值，被判负，返回0
            return 0
        #若被某一层layer判正，则进入下一个layer的分类
    #倍所有的layer判正，则被cascade判正
    return 1

        
def write_cascade_to_file( cas ):
    file = open('Cascade.txt','w')
    file.write(cas.toString())
    file.close()
        
        
def read_cascade_from_file( fileName='Cascade.txt' ):
    file = open(fileName , 'r')
    cas = Cascade()
    
    #读取Cascade中的Layer个数
    cas.numberOfLayers = int(file.readline().strip())
    cas.layerList = []
    
    #进行每一个Layer的读取
    for i in range(cas.numberOfLayers):
        layer = Layer()
        layer.numberOfFeatures = int(file.readline().strip())
        layer.seFeatureList = []
        
        #进行Layer中每一个seFeature的读取
        for j in range(layer.numberOfFeatures):
            line = file.readline()
            lineList = line.strip().split('\t')
            f = SeFeature( float(lineList[0]) , \
                           Rectangle( int(lineList[1]), int(lineList[2]), int(lineList[3]), int(lineList[4]), int(lineList[5]) ),
                           int(lineList[6]), float(lineList[7]) )
            layer.seFeatureList.append(f)
        
        #读取Layer的阈值
        layer.thres = float(file.readline().strip())
        
        #完成layer的读取后添加进cascade
        cas.layerList.append(layer)
    
    return cas
    

    

    
    
    