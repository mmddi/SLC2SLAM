#surf descriptor
# Jorge Guevara

import cv2
import numpy as np 


#doc at http://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#gsc.tab=0
def describeSURF( image):
    surf = cv2.xfeatures2d.SURF_create()
    # it is better to have this value between 300 and 500
    surf.setHessianThreshold(400)
    kp, des = surf.detectAndCompute(image,None)
    return kp,des

#doc at http://docs.opencv.org/master/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html#gsc.tab=0
def describeSIFT( image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    #draw keypoints
    #import matplotlib.pyplot as plt		
    #img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    #plt.imshow(img2),plt.show()
    return kp,des

def describeORB( image):
    #An efficient alternative to SIFT or SURF
    #doc http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    #ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor 
    #with many modifications to enhance the performance
    #print("Image shape:", image.shape)
    #print("Number of channels:", image.shape[2] if len(image.shape) == 3 else 1)
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray_image = image
    else:
        # 将彩色图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print("******Image shape:", image.shape)
    #print("*******Number of channels:", image.shape[2] if len(image.shape) == 3 else 1)

    orb=cv2.ORB_create()
      #将彩色图像转为了灰度图像

    kp, des=orb.detectAndCompute(gray_image,None)
    return kp,des





 

