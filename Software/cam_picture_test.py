#!/usr/bin/python3

import sys
import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

win0 = "Camera"

cv2.namedWindow(win0, cv2.WINDOW_NORMAL) 

# initialize the camera 
camera = PiCamera(resolution=(1280, 720), framerate=10)

# set camera parameters
camera.iso = 100
time.sleep(2)
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g

# grab a reference to the raw camera capture
rawCapture = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.5)

# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# display the image on screen and wait for a keypress
cv2.imshow(win0, image)
cv2.waitKey(0)

