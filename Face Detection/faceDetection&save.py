# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:40:08 2023

@author: USER
"""

import cv2
import os

dataset = "facedata"
name = "champ"

path = os.path.join(dataset,name)

if not os.path.isdir(path):
    os.mkdir(path)


(width,height) = (130,100)

algo = 'haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(algo)


cam = cv2.VideoCapture(0)
count = 1
text = "No Person Detected"
while count < 2:    
    _,img = cam.read()
    grayImg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
     
    for(x,y,w,h) in face:
        cv2.rectangle(img, (x,y) , (x+w,y+h) , (0,255,0) , 2)
        faceonly = grayImg[y:y+h , x:x+w]
        resizeImg = cv2.resize(faceonly , (width , height))
        cv2.imwrite("%s/%s.jpg" %(path,count),resizeImg)
        count +=1
        text="Person Detected"
    cv2.imshow("FaceDetection", img)
    print(text)
    
   
    key = cv2.waitKey(10)
    if key == 27:
        break
print("Image Capture Successful")
cam.release()
cv2.destroyAllWindows()










        