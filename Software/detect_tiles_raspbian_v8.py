#!/usr/bin/python3
import sys
import cv2
import numpy as np

### Available images to be processed

filename1  = './../Kilian_examples/ko_top_corr2.jpg'                  # not ok
filename2  = './../Kilian_examples/hrt_top.jpg'                       # ok without autoCurves1
filename3  = './../Kilian_examples/tests/p1.jpg'                      # ok without autoCurves1
filename4  = './../Kilian_examples/tests/p1_gaps.jpg'                 # not ok
filename5  = './../Kilian_examples/tests/p2_landscape.jpg'            # not ok
filename6  = './../Kilian_examples/tests/p2_landscape_far_blue.jpg'   # not ok
filename7  = './../Kilian_examples/tests/p2_landscape_far_bright.jpg' # not ok
filename8  = './../Kilian_examples/tests/p2_landscape_gaps.jpg'       # ok without autoCurves1
filename9  = './../Kilian_examples/tests/p2_rot.jpg'                  # not ok
filename10 = './../Kilian_examples/tests/p2_rot180_dist_gaps.jpg'     # not ok
filename11 = './../Kilian_examples/tests/p2_spaces.jpg'
filename12 = './../Kilian_examples/tests/p2_spaces_blue1.jpg'
filename13 = './../Kilian_examples/tests/p2_spaces_blue2.jpg'
filename14 = './../Kilian_examples/ym_top_corr.jpg'                   # not ok

useFile = filename2        # Select which image to use

use_maskDetection = "HSV"  # How to detect the black and white masks for the image contour: HSV [recommended], RGB or LUM (luminocity on grayscale)
use_autoCurves1 = False    # If True [recommended for some images], program will perform histogram autotune on the original image to determine the black and white masks
use_autoCurves2 = True     # If True [recommended], program will perform contrast adjustment on the cropped grayscale image 
pxSize = 200               # Length (px) of a machine-produced tile's side edge
numTilesEach = 5	       # Number of each type of tile in the game (only for plausibility check)

### Upper thresholds for black/white/medium detection

thr_black = 90
thr_center = 180
thr_white = 255

### Window objects names

win0 = "0 Original"
win1 = "1 Mask white"
win2 = "2 Mask black"
win3 = "3 Mask combined"
win4 = "4 Crop area"
win5 = "5 Grayscale cropped"
win6 = "6 Result"

cv2.namedWindow(win0, cv2.WINDOW_NORMAL) 
cv2.namedWindow(win1, cv2.WINDOW_NORMAL) 
cv2.namedWindow(win2, cv2.WINDOW_NORMAL) 
cv2.namedWindow(win3, cv2.WINDOW_NORMAL) 
cv2.namedWindow(win4, cv2.WINDOW_NORMAL) 
cv2.namedWindow(win5, cv2.WINDOW_NORMAL) 
cv2.namedWindow(win6, cv2.WINDOW_NORMAL) 

### Define all existing tiles

baseSignatures = np.empty( (40),dtype=float).reshape(10,4)
numRotations = np.empty( (10),dtype="uint8")
bitmaps = np.zeros(10*pxSize*pxSize).reshape(10,pxSize,pxSize)

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

### Define necessary parameters for the tiles
				
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

createBitmaps()
defineRotations()
defineBaseSignatures()

### Create tile objects with parameters

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
		

givenTiles = np.empty( (10), dtype=Tile)
for i in range(10):
	givenTiles[i] = Tile.specific(i,baseSignatures[i],numRotations[i],bitmaps[i],False,0)
givenTiles[6].needsExtraProbe = True
givenTiles[6].extraProbeValue = 1.0



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
		
		#cv2.imshow("Original",self.img_original)
		cv2.imshow("Result",self.img_result)
		cv2.waitKey(0)
		
	
resultObject = Result()





### Read image

img = cv2.imread(useFile)
resultObject.img_original = img.copy()



### Pre-cropping (optional)

#img = img[y0:y1, x0:x1]



### Determine image size

h,w,_ = img.shape
print(str(w)+" x "+str(h)+" px")
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
	
	
########################################################################
### FIND OUTER CONTOUR #################################################
########################################################################	
	

### Auto-curves 1: histogram optimization

if use_autoCurves1 == True:
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * float(hist.max()) / cdf.max()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	img = cdf[img]

img2 = cv2.resize(img, (w2,h2))
cv2.imshow(win0,img2)

### Create masks: preparation

kernel1 = np.ones((4,4),np.uint8)
kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((4,4),np.uint8)
mask_detect = img.copy()
mask_white = np.empty([h,w])
mask_black = np.empty([h,w])

### Create a mask for white and another one for black pixels and remove the noise

if use_maskDetection == "HSV":
	mask_detect = cv2.cvtColor(mask_detect, cv2.COLOR_RGB2HSV)
	
	mask_detect_small = cv2.resize(mask_detect, (w2,h2))
	cv2.imshow("HSV",mask_detect_small)
	
	if use_autoCurves1 == True:
		mask_white = cv2.inRange(mask_detect, (0, 0, 209), (180, 255, 255))
		mask_black = cv2.inRange(mask_detect, (0, 0, 0), (180, 255, 87))	
	else:
		mask_white = cv2.inRange(mask_detect, (0, 0, 189), (153, 14, 255))
		mask_black = cv2.inRange(mask_detect, (0, 0, 0), (180, 90, 66))
		
elif use_maskDetection == "LUM":
	mask_detect = cv2.cvtColor(mask_detect, cv2.COLOR_RGB2GRAY)
	mask_white = cv2.inRange(mask_detect, 160,255)
	mask_black = cv2.inRange(mask_detect, 0,55)
	
elif use_maskDetection == "RGB":	
	white_lowRGB = np.array([150,150,150])
	white_highRGB = np.array([255,255,255])	
	#white_maxVariances_RGB = np.array([200,20,20])
	black_lowRGB = np.array([0,0,0])
	black_highRGB = np.array([85,77,68])	
	black_maxVariances_RGB = np.array([15,5,5])
	
	mask_R_inRange_white = mask_detect[:,:,0] # select red values in range for white	
	mask_R_inRange_white = (mask_R_inRange_white >= white_lowRGB[0]) & (mask_R_inRange_white <= white_highRGB[0])
	mask_G_inRange_white = mask_detect[:,:,1] # select green values in range for white	
	mask_G_inRange_white = (mask_G_inRange_white >= white_lowRGB[1]) & (mask_G_inRange_white <= white_highRGB[1])
	mask_B_inRange_white = mask_detect[:,:,2] # select blue values in range for white	
	mask_B_inRange_white = (mask_B_inRange_white >= white_lowRGB[2]) & (mask_B_inRange_white <= white_highRGB[2])
	
	mask_white = mask_R_inRange_white * mask_G_inRange_white * mask_B_inRange_white
	mask_white = 255*(mask_white.astype("uint8"))
	
	mask_R_inRange_black = mask_detect[:,:,0] # select red values in range for black	
	mask_R_inRange_black = (mask_R_inRange_black >= black_lowRGB[0]) & (mask_R_inRange_black <= black_highRGB[0])
	mask_G_inRange_black = mask_detect[:,:,1] # select green values in range for black	
	mask_G_inRange_black = (mask_G_inRange_black >= black_lowRGB[1]) & (mask_G_inRange_black <= black_highRGB[1])
	mask_B_inRange_black = mask_detect[:,:,2] # select blue values in range for black	
	mask_B_inRange_black = (mask_B_inRange_black >= black_lowRGB[2]) & (mask_B_inRange_black <= black_highRGB[2])
	mask_black = mask_R_inRange_black * mask_G_inRange_black * mask_B_inRange_black
	#mask_black = 255*(mask_black.astype("uint8"))
	
	mask_detect = cv2.blur(mask_detect,(50,50))
	#cv2.imshow("Img blur",cv2.resize(mask_detect,(w2,h2)))
	
	averages = np.mean(mask_detect,axis=2).astype("uint8")		# grayscale image
	
	mask_R_offVarRange_black = mask_detect[:,:,0]
	mask_R_offVarRange_black = abs(mask_R_offVarRange_black - averages) > black_maxVariances_RGB[0]
	#mask_R_offVarRange_black = 255*(mask_R_offVarRange_black.astype("uint8"))
	
	mask_G_offVarRange_black = mask_detect[:,:,1]
	mask_G_offVarRange_black = abs(mask_G_offVarRange_black - averages) > black_maxVariances_RGB[1]
	#mask_G_offVarRange_black = 255*(mask_G_offVarRange_black.astype("uint8"))
	
	mask_B_offVarRange_black = mask_detect[:,:,2]
	mask_B_offVarRange_black = abs(mask_B_offVarRange_black - averages) > black_maxVariances_RGB[2]
	#mask_B_offVarRange_black = 255*(mask_B_offVarRange_black.astype("uint8"))
	
	
	
	
	#cv2.imshow("Red in not black",cv2.resize(mask_R_offVarRange_black,(w2,h2)))
	#cv2.imshow("Green in not black",cv2.resize(mask_G_offVarRange_black,(w2,h2)))
	#cv2.imshow("Blue in not black",cv2.resize(mask_B_offVarRange_black,(w2,h2)))
	
	mask_highVar = mask_R_offVarRange_black * mask_G_offVarRange_black * mask_B_offVarRange_black
	print(mask_highVar)
	#print(np.count_nonzero(mask_highVar == True))
	#mask_highVar = 255*(mask_highVar.astype("uint8"))
	#cv2.imshow("Colorful",cv2.resize(mask_highVar,(w2,h2)))
	
	mask_lowVar = np.bitwise_not(mask_highVar)
	#mask_lowVar = 255*(mask_lowVar.astype("uint8"))
	#cv2.imshow("Colorless",cv2.resize(mask_lowVar,(w2,h2)))
	
	mask_black = mask_black * mask_lowVar
	mask_black = 255*(mask_black.astype("uint8"))
	
	cv2.imshow("White",cv2.resize(mask_white,(w2,h2)))
	cv2.imshow("Black",cv2.resize(mask_black,(w2,h2)))
	cv2.waitKey(0)
	#print(mask_R_inRange_white)
	#print(np.count_nonzero(mask_R_inRange_white == True))
	#print(mask_R_inRange_white)
	#print(mask_detect)
	#print(mask_R_inRange_white)

mask_white = cv2.dilate(mask_white,kernel1,iterations = 1)
mask_white = cv2.erode(mask_white,kernel1,iterations = 1)
mask_black = cv2.erode(mask_black,kernel1,iterations = 1)
mask_black = cv2.dilate(mask_black,kernel1,iterations = 1)

mw2 = cv2.resize(mask_white.copy(), (round(w2/2),round(h2/2)))
mb2 = cv2.resize(mask_black.copy(), (round(w2/2),round(h2/2)))
cv2.imshow(win1,mw2)
cv2.imshow(win2,mb2)

### Combine the two masks into one and remove the remaining noise

mask_crop = mask_black+mask_white
mask_crop = mask_crop.clip(0,255).astype("uint8")
mask_crop = cv2.erode(mask_crop,kernel2,iterations = 1)
mask_crop = cv2.dilate(mask_crop,kernel2,iterations = 1)
mask_crop = cv2.dilate(mask_crop,kernel3,iterations = 1)
mask_crop = cv2.erode(mask_crop,kernel3,iterations = 1)

mc2 = cv2.resize(mask_crop, (round(w2/2),round(h2/2)))
cv2.imshow(win3,mc2)

### Detect the full square

_,thresh = cv2.threshold(mask_crop,254,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours: "+str(len(contours)))

if len(contours) != 0:
	cnt = max(contours, key = cv2.contourArea)
	x,y,rw,rh = cv2.boundingRect(cnt)
	mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
	
########################################################################
### CROP AND ANALYZE ###################################################
########################################################################

	### Crop image to detected size

	img_cropped = img[y:y+rh, x:x+rw]
	
	### Draw grid (for preview purposes only)
	
	#mask_white_bgr = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
	gridsize = np.array([(rw/5),(rh/5)], dtype=float)		
	img_raster = img.copy()	
	cv2.rectangle(img_raster,(x,y),(x+rw,y+rh), (255,255,0), 10, 1)
	
	'''	
	for col in range(6):
		start_x = x + round(col*gridsize[0])
		start_y = y
		end_x = start_x
		end_y = start_y + rh
		cv2.line(img_raster, (start_x, start_y), (end_x, end_y), (255, 255, 0), 4, 1)		
	for row in range(6):
		start_x = x
		start_y = y + round(row*gridsize[1])
		end_x = start_x + rw
		end_y = start_y
		cv2.line(img_raster, (start_x, start_y), (end_x, end_y), (255, 255, 0), 4, 1)	
	'''
	
	img_raster = cv2.resize(img_raster, (round(w2/2),round(h2/2)))
	cv2.imshow(win4,img_raster)	
	
	### Create 25 tiles and probe each of them in 5 places, then save results in signatures[5][5][5]
	
	# first dimension (5) : columns
	# second dimension (5): rows
	# third dimension (5) : probed results: 0: left, 1: right, 2: top, 3: bottom, 4: extra top-left (for tile 6)
	averages = np.zeros(125).reshape(5,5,5)
	signatures = np.zeros(125).reshape(5,5,5)
	identifiers = np.zeros(50).reshape(5,5,2).astype("uint8")
	confidences = np.zeros(5*25).reshape(5,5,5).astype(float)
	identified = np.zeros(25).reshape(5,5).astype(bool)
	avgConfidences = np.zeros(25).reshape(5,5).astype(float)
	
	rectHalfLong = np.floor(gridsize[0] * 0.5 * 0.5).astype("uint8")
	rectHalfShort = np.floor(rectHalfLong/11).astype("uint8")
	print("Probe rect: "+str(rectHalfLong)+" x "+str(rectHalfShort)+" px")
	
	img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
	
	if use_autoCurves2 == True:
		# eliminating middle tones
		img_cropped_gray[img_cropped_gray < thr_black] = 0
		img_cropped_gray[img_cropped_gray >= thr_black] = 255	
			
	img_cropped_gray_disp = img_cropped_gray.copy()
	img_cropped_gray_disp = cv2.cvtColor(img_cropped_gray_disp, cv2.COLOR_GRAY2BGR)
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
	img_cropped_gray_disp = cv2.resize(img_cropped_gray_disp, (round(rw*3),round(rh*3)))		
	cv2.imshow(win5,img_cropped_gray_disp)
	
	numIdentified = 0
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
			
			
			
			
			
			
			
########################################################################
### IDENTIFY ###########################################################
########################################################################
			
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
										print("a Found! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
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
										print("b1 Found! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
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
											print("b2 Found! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
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
										print("c Found! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
										identifiers[col][row][0] = givenTile.num
										identifiers[col][row][1] = rot
										signatures[col][row][i] = 1
										break
									else:
										signatures[col][row][i] = 1
								
					
		print("\nIdentified tiles (after pass 2): "+str(numIdentified))
	
	
	
	### Check if the result may be good
	
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
	
	
	
	
	resultObject.detection_avgConfidence = avgConfidences.sum() / 25
	resultObject.detection_identified = identified.copy()
	resultObject.detection_success = plausible


	
########################################################################
### OUTPUT IMAGE #######################################################
########################################################################	
				
			
			
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
	
	cv2.imshow(win6,outputImage)
	key = cv2.waitKey(0)


	
	cv2.destroyAllWindows()
	resultObject.printAll()
	
	
	
	
else:
	print("No contour found.")
	cv2.destroyAllWindows()
	exit(2)

cv2.destroyAllWindows()




