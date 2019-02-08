#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:14:26 2019

@author: rohitgupta
"""

import imutils
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

if not args.get("video",False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])
    
while True:
    (grabbed, frame) = camera.read()
    
    # if we are viewing a video and we did not grab a frame, we've reached the 
    # end of the video
    if args.get("video") and not grabbed:
        break
    
    # resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    
    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    
    # These erosions and dilations will help remove the small false-positive 
    # skin regions in the image.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    
    # blur the mask to help remove noise, then apply the
	# mask to the frame
    skinmask = cv2.GaussianBlur(skinMask, (5,5), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    
    cv2.imshow("images", np.hstack([frame, skin]))
    
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.distroyAllWindows()