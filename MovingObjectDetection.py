# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 22:19:29 2023

@author: USER
"""

import cv2
import imutils
img = cv2.imread("Ai.png")
resize = imutils.resize(img,width=200)
cv2.imwrite("resizedimg.png",resize) 

