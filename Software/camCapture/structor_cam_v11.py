#!/usr/bin/python3
import sys
import cv2
import numpy as np
import time
import math
from picamera.array import PiRGBArray
from picamera import PiCamera


### Operating mode

use_crop_sliders = True      # If True, cropping area may be manually defined using sliders
static_crop_margins = np.array([0,0,0,0],dtype=int) # left, top, right, bottom
use_autoCurves1 = False    # If True [recommended for some images], program will perform histogram autotune on the original image
use_imageOutput = False     # Activate or deactivate output images and cv2.waitKey()

### Settings for detection of tile types

use_autoCurves2 = True     # If True [recommended], program will perform contrast adjustment on the cropped grayscale image 
numTilesEach = 5	       # Number of each type of tile in the game (only for plausibility check)

### Settings for generated output image

pxSize = 200               # Length (px) of a machine-produced tile's side edge

### Upper thresholds for black/white/medium detection

thr_black = 90
thr_center = 180
thr_white = 255


######################################################

state = 0
camera = None
rawCapture = None
img = None
img_cropped = None
w = None
h = None
w2 = None
h2 = None
x = None
y = None
rw = None
rh = None
rectHalfLong = 0
rectHalfShort = 0
mult = 1.0
numIdentified = 0
gridsize = None
givenTiles = None
plausible = False

### Window objects names

win0 = "Original"	
win1 = "Crop area"
win2 = "Result"	

### Crop sliders

slider_crop_left_name = 'Crop left'
slider_crop_top_name = 'Crop top'
slider_crop_right_name = 'Crop right'
slider_crop_bottom_name = 'Crop bottom'

### Other



def left_margin_trackbar(val):
    global static_crop_margins
    static_crop_margins[0] = val
    cv2.setTrackbarPos(slider_crop_left_name, win0, static_crop_margins[0])
def top_margin_trackbar(val):
    global static_crop_margins
    static_crop_margins[1] = val
    cv2.setTrackbarPos(slider_crop_top_name, win0, static_crop_margins[1])
def right_margin_trackbar(val):
    global static_crop_margins
    static_crop_margins[2] = val
    cv2.setTrackbarPos(slider_crop_right_name, win0, static_crop_margins[2])
def bottom_margin_trackbar(val):
    global static_crop_margins
    static_crop_margins[3] = val
    cv2.setTrackbarPos(slider_crop_bottom_name, win0, static_crop_margins[3])
    



### Tile class

class Tile:
	num = 0
	baseSignature = np.zeros(4) # left, right, top, bottom
	needsExtraProbe = False
	extraProbeValue = 0
	numRotations = 0
	bitmap = np.zeros(pxSize*pxSize).reshape(pxSize,pxSize)
	 
	brightness = 0.0
	 
	def __init__(self, _num):
		self.num = _num
	 
	@classmethod
	def specific(cls,_num,_baseSignature,_numRotations,_bitmap,_needsExtraProbe,_extraProbeValue):
		obj = cls.__new__(cls)
		obj.num = _num
		obj.baseSignature = _baseSignature
		obj.numRotations = _numRotations
		obj.bitmap = _bitmap
		obj.needsExtraProbe = _needsExtraProbe
		obj.extraProbeValue = _extraProbeValue
		obj.brightness = obj.calcBrightness()
		return obj
		
	def print(self):
		print("Tile "+str(self.num))
		print("  Base signature       : "+str(self.baseSignature))
		print("  Distinct orientations: "+str(self.numRotations+1))
		print("  Needs extra probe    : "+str(self.needsExtraProbe))
		print("  Extra probe value    : "+str(self.extraProbeValue))
		
	def isMe(self,signature):
		thisSignature = self.baseSignature.copy()
		if (np.isclose(signature[:4],thisSignature)).sum() == 4:
			if self.needsExtraProbe == True:
				if np.isclose(signature[4],self.extraProbeValue) == 1:
					return 0
				else:
					return 1
			return 0
		for rot in range(1,self.numRotations+1):
			thisSignature = self.rotate(thisSignature)
			if (np.isclose(signature[:4],thisSignature)).sum() == 4:
				return rot
		return -1
	
	def rotate(self, l, y=1):
		return np.roll(l,y)	
		
	def rotImage(self,rot):
		rotBitmap = self.bitmap.copy()
		return np.rot90(rotBitmap,-rot)
		
	def calcBrightness(self):
		return self.bitmap.sum() / (pxSize*pxSize*255)


### Result class

class Result:
	img_original = np.empty( (1), dtype="uint8" )
	img_original_pxSize = np.zeros((2),dtype=int)
	img_result =  np.empty( (1), dtype="uint8" )
	result_signature = np.zeros((2*25),dtype="uint8").reshape(5,5,2)	
	detection_tiles_probes = np.zeros((5*25),dtype="uint8").reshape(5,5,5)
	detection_avgConfidence = 0.0
	detection_success = False
	detection_confidences = np.zeros((5*25),dtype=float).reshape(5,5,5)
	detection_identified = np.zeros((25),dtype=bool).reshape(5,5)
	
	def __init__(self):
		pass
	
	@classmethod
	def defineAll(cls, _img_original, _img_original_pxSize, _img_result, _result_signature, _detection_tiles_probes, _detection_avgConfidence, _detection_success, _detection_confidences, _detection_identified):
		obj = cls.__new__(cls)
		obj.img_original = _img_original
		obj.img_original_pxSize = _img_original_pxSize
		obj.img_result = _img_result
		obj.result_signature = _result_signature
		obj.detection_tiles_probes = _detection_tiles_probes
		obj.detection_avgConfidence = _detection_avgConfidence
		obj.detection_success = _detection_success
		obj.detection_confidences = _detection_confidences
		obj.detection_identified = _detection_identified
		return obj
		
	def printAll(self):
		print("Original image pixel size: (w x h)"+str(self.img_original_pxSize[0])+" x "+str(self.img_original_pxSize[1]))
		print("Probed values:\n  "+str(self.detection_tiles_probes))
		print("Probes confidences:\n  "+str(self.detection_confidences))
		print("Tiles identified:\n  "+str(self.detection_identified))
		print("Overall avg. confidence: "+str(self.detection_avgConfidence))
		
		print("\nResult signature:\n  "+str(self.result_signature))
		print("Detection success: "+str(self.detection_success))
		



def initialize():
	global camera
	global rawCapture
	
	
	
	
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
	#print(type(rawCapture))

	# allow the camera to warmup
	time.sleep(0.5)		
		
	if use_crop_sliders == True and use_imageOutput == True:
		cv2.createTrackbar(slider_crop_left_name, win0 , 0, (int)(1280/2), left_margin_trackbar)
		cv2.createTrackbar(slider_crop_top_name, win0 , 0, (int)(720/2), top_margin_trackbar)
		cv2.createTrackbar(slider_crop_right_name, win0 , 0, (int)(1280/2), right_margin_trackbar)
		cv2.createTrackbar(slider_crop_bottom_name, win0 , 0, (int)(720/2), bottom_margin_trackbar)
		
	
	



def createTiles():
	global baseSignatures
	global numRotations
	global bitmaps
	global pxSize
	global givenTiles

	### Define all existing tiles

	baseSignatures = np.empty( (40),dtype=float).reshape(10,4)
	numRotations = np.empty( (10),dtype="uint8")
	bitmaps = np.zeros(10*pxSize*pxSize).reshape(10,pxSize,pxSize)
	
	createBitmaps()
	defineRotations()
	defineBaseSignatures()
	
	givenTiles = np.empty( (10), dtype=Tile)
	for i in range(10):
		givenTiles[i] = Tile.specific(i,baseSignatures[i],numRotations[i],bitmaps[i],False,0)
	givenTiles[6].needsExtraProbe = True
	givenTiles[6].extraProbeValue = 1.0


def createBitmaps():
	global bitmaps
	for r in range(pxSize):
		for c in range(pxSize):
			# upper half:
			if r < np.floor(pxSize/2):
				bitmaps[5][r][c] = 255
				bitmaps[9][r][c] = 255
				if c < np.floor(pxSize/2):
					bitmaps[2][r][c] = 255
					bitmaps[4][r][c] = 255
					bitmaps[6][r][c] = 255
					if r <= c:
						bitmaps[1][r][c] = 255
						bitmaps[3][r][c] = 255	
					if c <= r:
						bitmaps[7][r][c] = 255					
				else:
					bitmaps[8][r][c] = 255	
					if r < pxSize - c:
						bitmaps[1][r][c] = 255
						bitmaps[3][r][c] = 255
						bitmaps[4][r][c] = 255
					if pxSize - c <= r:
						bitmaps[7][r][c] = 255
			# lower half:
			else:
				bitmaps[7][r][c] = 255
				bitmaps[8][r][c] = 255
				bitmaps[9][r][c] = 255
				if c < np.floor(pxSize/2):
					if pxSize - c <= r:
						bitmaps[3][r][c] = 255
					if r < pxSize - c:
						bitmaps[4][r][c] = 255
				else:
					bitmaps[6][r][c] = 255
					if c <= r:
						bitmaps[3][r][c] = 255	

def defineRotations():
	global numRotations
	numRotations[0] = 0
	numRotations[1] = 3
	numRotations[2] = 3
	numRotations[3] = 1
	numRotations[4] = 3
	numRotations[5] = 3
	numRotations[6] = 1
	numRotations[7] = 3
	numRotations[8] = 3
	numRotations[9] = 0
				
def defineBaseSignatures():
	global baseSignatures
	baseSignatures[0] = [0, 0, 0, 0]	
	baseSignatures[1] = [0, 1, 0, 0]
	baseSignatures[2] = [0.5, 0.5, 0, 0]
	baseSignatures[3] = [0, 1, 0, 1]
	baseSignatures[4] = [1, 1, 0, 0]
	baseSignatures[5] = [0.5, 1, 0.5, 0]
	baseSignatures[6] = [0.5, 0.5, 0.5, 0.5]
	baseSignatures[7] = [1, 0, 1, 1]
	baseSignatures[8] = [0.5, 0.5, 1, 1]
	baseSignatures[9] = [1, 1, 1, 1]	
	
	
def getImage():
	global img
	global resultObject
	global w, h, w2, h2
	global camera
	global rawCapture
	global mult
	
	### Get camera image	
	
	camera.capture(rawCapture, format="bgr")
	img = rawCapture.array
	
	resultObject.img_original = img.copy()
	
	### Determine image size

	h,w,_ = img.shape
	print("Image: "+str(w)+" x "+str(h)+" px")
	resultObject.img_original_pxSize = np.array([w,h])

	### Resize (for display purposes only)

	w2 = w
	h2 = h
	mult = 1.0
	if h > 800 or w > 2000:
		longest = max(w,h)
		if(w < h):
			if longest == h:
				mult = 800 / longest
			else:
				mult = 2000 / longest
		else:
			mult = 800 / h
		w2 = round(mult * w)
		h2 = round(mult * h)


def optimizeImage():
	global img
	
	### Auto-curves 1: histogram optimization

	if use_autoCurves1 == True:
		hist,bins = np.histogram(img.flatten(),256,[0,256])
		cdf = hist.cumsum()
		cdf_normalized = cdf * float(hist.max()) / cdf.max()
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')
		img = cdf[img]

	

			

def cropImage():
	global img_cropped, img_cropped_gray, img_cropped_gray_disp
	
	### Crop image to selected or detected size

	img_cropped = img[y:y+rh, x:x+rw]
	img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
	img_cropped_gray_disp = img_cropped_gray.copy()
	img_cropped_gray_disp = cv2.cvtColor(img_cropped_gray_disp, cv2.COLOR_GRAY2BGR)
	
	### Show image with grid
	
	for col in range(6):
		start_x = int(round(col*gridsize[0]))
		start_y = 0
		end_x = start_x
		end_y = rh
		cv2.line(img_cropped_gray_disp, (start_x, start_y), (end_x, end_y), (255, 255, 0), 6, 1)		
	for row in range(6):
		start_x = 0
		start_y = int(round(row*gridsize[1]))
		end_x = rw
		end_y = start_y
		cv2.line(img_cropped_gray_disp, (start_x, start_y), (end_x, end_y), (255, 255, 0), 6, 1)	
		
	if use_imageOutput == True:
		cv2.namedWindow(win1, cv2.WINDOW_NORMAL) 	
		cv2.imshow(win1,img_cropped_gray_disp)
		cv2.waitKey(1)

	
def showOriginalImage():
	global x, y, rw, rh, gridsize
	global img2
	
	if use_imageOutput == True:
		cv2.namedWindow(win0, cv2.WINDOW_NORMAL) 
		
	y = static_crop_margins[1]
	rh = h - static_crop_margins[1] - static_crop_margins[3]
	x = static_crop_margins[0]
	rw = w - static_crop_margins[2] - static_crop_margins[0]
	gridsize = np.array([(round(mult*rw/5)),(round(mult*rh/5))], dtype=int)	
		
	while(use_imageOutput == True):
		y = static_crop_margins[1]
		rh = h - static_crop_margins[1] - static_crop_margins[3]
		x = static_crop_margins[0]
		rw = w - static_crop_margins[2] - static_crop_margins[0]

		img2 = cv2.resize(img, (w2,h2)) # Preview resized
		
		### Draw border and grid (for preview purposes only)
		
		left_start  = np.array([round(mult*x),round(mult*y)],dtype=int)
		left_end    = np.array([round(mult*x),round(mult*(y + rh))],dtype=int)
		right_start = np.array([round(mult*(x + rw)),round(mult*y)],dtype=int)
		right_end   = np.array([round(mult*(x + rw)),round(mult*(y + rh))],dtype=int)
		cv2.line(img2, (left_start[0],left_start[1]), (left_end[0],left_end[1]), (255, 255, 0), 4, 1)	    # left border
		cv2.line(img2, (left_start[0],left_start[1]), (right_start[0],right_start[1]), (255, 255, 0), 4, 1)	# top border
		cv2.line(img2, (right_start[0],right_start[1]), (right_end[0],right_end[1]), (255, 255, 0), 4, 1)	# right border
		cv2.line(img2, (left_end[0],left_end[1]), (right_end[0],right_end[1]), (255, 255, 0), 4, 1)	        # bottom border
		
		gridsize = np.array([(round(mult*rw/5)),(round(mult*rh/5))], dtype=int)		
		#print("gridsize: "+str(gridsize))
		
		for col in range(1,5):
			line_start = np.array([left_start[0] + col*gridsize[0],left_start[1]],dtype=int)
			line_end   = np.array([left_start[0] + col*gridsize[0],left_end[1]],dtype=int)			
			cv2.line(img2, (line_start[0], line_start[1]), (line_end[0], line_end[1]), (255, 255, 0), 4, 1)		
		
		for row in range(1,5):
			line_start = np.array([left_start[0],left_start[1] + row*gridsize[1]],dtype=int)
			line_end   = np.array([right_start[0],left_start[1] + row*gridsize[1]],dtype=int)			
			cv2.line(img2, (line_start[0], line_start[1]), (line_end[0], line_end[1]), (255, 255, 0), 4, 1)				
					
		cv2.imshow(win0,img2)		
		key = cv2.waitKey(1)
		if key == ord('q') or key == 27:
			break	
	
	
	
	
def initializeAnalysis():
	global averages, signatures, identifiers, confidences, identified, avgConfidences, rectHalfLong, rectHalfShort
	# first dimension (5) : columns
	# second dimension (5): rows
	# third dimension (5) : probed results: 0: left, 1: top, 2: right, 3: bottom, 4: extra top-left (for tile 6)
	averages = np.zeros(125).reshape(5,5,5)
	signatures = np.zeros(125).reshape(5,5,5)
	identifiers = np.zeros(50).reshape(5,5,2).astype("uint8")
	confidences = np.zeros(5*25).reshape(5,5,5).astype(float)
	identified = np.zeros(25).reshape(5,5).astype(bool)
	avgConfidences = np.zeros(25).reshape(5,5).astype(float)
	print("Gridsize: "+str(gridsize[0])+" x "+str(gridsize[1]))
	
	gridShortest = min(gridsize[0],gridsize[1])
	rectHalfLong = np.floor(gridShortest * 0.5 * 0.5).astype("uint8")
	rectHalfShort = np.floor(rectHalfLong/11).astype("uint8")
	print("Probe rect: "+str(2*rectHalfLong)+" x "+str(2*rectHalfShort)+" px")


def optimizeCropImage():
	global img_cropped_gray
	if use_autoCurves2 == True:
		# eliminating middle tones
		img_cropped_gray[img_cropped_gray < thr_black] = 0
		img_cropped_gray[img_cropped_gray >= thr_black] = 255	
		
def identifyTiles():
	global signatures, confidences, averages, avgConfidences, resultObject, numIdentified
	
	print("Identifying tiles (pass 1)...")
	
	for col in range(5):
		for row in range(5):
			start_x = int(round(col*gridsize[0]))
			start_y = int(round(row*gridsize[1]))
			end_x = start_x + int(round(gridsize[0]))
			end_y = start_y + int(round(gridsize[1]))		
			
			tile = img_cropped_gray[start_y:end_y, start_x:end_x]
			
			sums = np.zeros(5)
			avgs = np.zeros(5)

			for c in range(int(2*rectHalfShort)):
				for r in range(int(2*rectHalfLong)):		
					val1 = tile[int(round(gridsize[1]/2-rectHalfLong)+r)][int(round(gridsize[0]/4-rectHalfShort)+c)]
					val3 = tile[int(round(gridsize[1]/2-rectHalfLong)+r)][int(round(3*gridsize[0]/4-rectHalfShort)+c)]
					sums[0] += val1
					sums[2] += val3
			
			for c in range(int(2*rectHalfLong)):
				for r in range(int(2*rectHalfShort)):					
					val2 = tile[int(round(gridsize[1]/4-rectHalfShort)+r)][int(round(gridsize[0]/2-rectHalfLong)+c)]
					val4 = tile[int(round(3*gridsize[1]/4-rectHalfShort)+r)][int(round(gridsize[0]/2-rectHalfLong)+c)]			
					sums[1] += val2
					sums[3] += val4
			
			for c in range(int(2*rectHalfShort)):
				for r in range(int(2*rectHalfShort)):
					val5 = tile[int(round(gridsize[1]/4-rectHalfShort)+r)][int(round(gridsize[0]/4-rectHalfShort)+c)]
					sums[4] += val5

			for i in range(4):
				avgs[i] = sums[i] / (4*rectHalfLong*rectHalfShort)
			avgs[4] = sums[4] / (4*rectHalfShort*rectHalfShort)
			# avgs[] now holds the five probe values for this tile

			resultObject.detection_tiles_probes[col][row][:] = avgs[:]
			#print(resultObject.detection_tiles_probes[col][row])

			for i in range(5):
				averages[col][row][i] = avgs[i]
				if avgs[i] <= thr_black:
					signatures[col][row][i] = 0						
					confidences[col][row][i] = (thr_black - avgs[i]) / thr_black
				elif avgs[i] <= thr_center:
					signatures[col][row][i] = 0.5
					if avgs[i] < 255/2:
						confidences[col][row][i] = ((255/2 - thr_black) - (255/2 - avgs[i])) / (255/2 - thr_black)
					else:
						confidences[col][row][i] = ((thr_center - 255/2) - (avgs[i] - 255/2)) / (thr_center - 255/2)
				elif avgs[i] <= thr_white:
					signatures[col][row][i] = 1
					confidences[col][row][i] = ((thr_white - thr_center) - (thr_white - avgs[i])) / (thr_white - thr_center)

			resultObject.detection_confidences[col][row][:] = confidences[col][row][:]					
			avgConfidences[col][row] = confidences[col][row].sum() / 5    
				
			### Identify tile type and orientation

			print("\nIdentifying tile (col,row): "+str(col)+","+str(row))			
			print("  Signature: "+str(signatures[col][row]))	
			print("  Detection confidence: "+str(avgConfidences[col][row]))		
			for givenTile in givenTiles:
				rot = givenTile.isMe(signatures[col][row])
				if rot > -1:
					numIdentified += 1
					identified[col][row] = True
					print("Found! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
					identifiers[col][row][0] = givenTile.num
					identifiers[col][row][1] = rot
					break
					
					
					
	print("\nIdentified tiles (after pass 1): "+str(numIdentified))		

	if numIdentified < 25:		
		print("Identifying tiles (pass 2)...")
		for col in range(5):
			for row in range(5):
				if identified[col][row] == False:	
					print("\nIdentifying remaining tile (col,row): "+str(col)+","+str(row))			
					print("  Signature: "+str(signatures[col][row]))	
					print("  Detection confidence: "+str(avgConfidences[col][row]))		
					for i in range(5):
						if confidences[col][row][i] <= 0.5:
							
							if signatures[col][row][i] == 0:
								signatures[col][row][i] = 0.5
								for givenTile in givenTiles:
									rot = givenTile.isMe(signatures[col][row])
									if rot > -1:
										numIdentified += 1
										identified[col][row] = True
										print("Found (case a)! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
										identifiers[col][row][0] = givenTile.num
										identifiers[col][row][1] = rot
										signatures[col][row][i] = 0
										break
									else:
										signatures[col][row][i] = 0
									
							elif signatures[col][row][i] == 0.5:
								signatures[col][row][i] = 0
								for givenTile in givenTiles:
									rot = givenTile.isMe(signatures[col][row])
									if rot > -1:
										numIdentified += 1
										identified[col][row] = True
										print("Found (case b1)! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
										identifiers[col][row][0] = givenTile.num
										identifiers[col][row][1] = rot
										signatures[col][row][i] = 0.5
										break
									else:
										signatures[col][row][i] = 1
										rot = givenTile.isMe(signatures[col][row])
										if rot > -1:
											numIdentified += 1
											identified[col][row] = True
											print("Found (case b2)! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
											identifiers[col][row][0] = givenTile.num
											identifiers[col][row][1] = rot
											signatures[col][row][i] = 0.5
											break
										else:
											signatures[col][row][i] = 0.5
											
							elif signatures[col][row][i] == 1:
								signatures[col][row][i] = 0.5
								for givenTile in givenTiles:
									rot = givenTile.isMe(signatures[col][row])
									if rot > -1:
										numIdentified += 1
										print("Found (case c)! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
										identifiers[col][row][0] = givenTile.num
										identifiers[col][row][1] = rot
										signatures[col][row][i] = 1
										break
									else:
										signatures[col][row][i] = 1								
					
		print("\nIdentified tiles (after pass 2): "+str(numIdentified))


def checkPlausibility():
	global plausible
	numUsed = np.zeros( (10) ).astype("uint8")
	plausible = True
	if numIdentified != 25:
		plausible = False
	for col in range(5):
		if plausible == False:
			break
		for row in range(5):
			numUsed[identifiers[col][row][0]] += 1
			if numUsed[identifiers[col][row][0]] > numTilesEach:
				plausible = False
				break
		
	print("\nResult might be okay: "+str(plausible))





#########################################################################################################################################################
###########################              ################################################################################################################
########################### MAIN PROGRAM ################################################################################################################
###########################              ################################################################################################################
#########################################################################################################################################################



def run():
	global resultObject
	global state
	
	state = 1	
	initialize() ### Initialize camera, output window for original and crop sliders
	
	state = 2
	createTiles() ### Create fundamental tile objects with parameters
	
	state = 3
	resultObject = Result() ### Create result object
	
	state = 4
	getImage() ### Take the image from file or camera
	
	state = 5
	optimizeImage() ### Apply auto curves 1
	
	state = 6
	showOriginalImage() ### Show original image (with sliders, if use_crop_sliders == True)

		
	########################################################################
	### CROP IMAGE AND IDENTIFY TILES ######################################
	########################################################################

	state = 7
	cropImage() ### Apply the crop to the original image

	state = 8
	initializeAnalysis() ### Initialze variables used for analysis

	state = 9
	optimizeCropImage() ### Apply auto curves 2 (eliminating middle tones)

	state = 10
	identifyTiles() ### Identify all 25 tiles

	state = 11
	checkPlausibility() ### Check if the result can even be plausible

	### Write into result object
				
	state = 12
	resultObject.detection_avgConfidence = avgConfidences.sum() / 25
	resultObject.detection_identified = identified.copy()
	resultObject.detection_success = plausible

		
	########################################################################
	### OUTPUT RESULT IMAGE ################################################
	########################################################################	
					
	state = 13			
				
	### Assemble the machine-made equivalent image

	outputImage = np.zeros(pxSize*pxSize*5*5).reshape(5*pxSize,5*pxSize)

	print("\nBrightness values:")
	for col in range(5):
		for row in range(5):		
			start_y = row*pxSize
			start_x = col*pxSize
			tileNum = identifiers[col][row][0]
			rotNum = identifiers[col][row][1]
			thisTile = givenTiles[tileNum].rotImage(rotNum).copy()			
			outputImage[start_y:start_y+pxSize, start_x:start_x+pxSize] = thisTile	
			print( str(givenTiles[tileNum].brightness) ,end=' ')

	print("\n")

	if plausible == True:
		print("\nStructor image signature:\n"+str(identifiers.flatten()))
		
	resultObject.result_signature = identifiers.copy()
	resultObject.img_result = outputImage.copy()
	
	if use_imageOutput == True:
		cv2.imshow(win2,outputImage)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
	state = 14
	
	resultObject.printAll()
	
	state = 15
	
if __name__ == "__main__":
	run()	
	
	




