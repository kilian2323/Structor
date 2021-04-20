#!/usr/bin/python3
import sys
import cv2
import numpy as np
import math

#np.set_printoptions(threshold=sys.maxsize)

pxSize = 100  # Length (px) of a machine-produced tile's side edge

baseSignatures = np.empty( (40),dtype=float).reshape(10,4)
numRotations = np.empty( (10),dtype="uint8")
bitmaps = np.zeros(10*pxSize*pxSize).reshape(10,pxSize,pxSize)

def createBitmaps():
	global bitmaps
	for r in range(pxSize):
		for c in range(pxSize):
			# upper half:
			if r < math.floor(pxSize/2):
				bitmaps[5][r][c] = 255
				bitmaps[9][r][c] = 255
				if c < math.floor(pxSize/2):
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
				if c < math.floor(pxSize/2):
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
			
		

createBitmaps()
defineRotations()
defineBaseSignatures()

'''
for i in range(10):
	cv2.imshow('Tile '+str(i),bitmaps[i])
key = cv2.waitKey(0)
cv2.destroyAllWindows()
'''

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
		return obj
		
	def print(self):
		print("Tile "+str(self.num))
		print("  Base signature       : "+str(self.baseSignature))
		print("  Distinct orientations: "+str(self.numRotations+1))
		print("  Needs extra probe    : "+str(self.needsExtraProbe))
		print("  Extra probe value    : "+str(self.extraProbeValue))
		
	def isMe(self,signature):
		#print("Is me?: "+str(self.num))		
		thisSignature = self.baseSignature.copy()
		#print("  My base signature: "+str(thisSignature))
		if (np.isclose(signature[:4],thisSignature)).sum() == 4:
			if self.needsExtraProbe == True:
				if np.isclose(signature[4],self.extraProbeValue) == 4:
					return 0
				else:
					return 1
			return 0
		for rot in range(1,self.numRotations+1):
			thisSignature = self.rotate(thisSignature)
			#print("  Rotated: "+str(thisSignature))
			#print("  Comparison result: "+str((np.isclose(compSignature,thisSignature)).sum()))
			if (np.isclose(signature[:4],thisSignature)).sum() == 4:
				return rot
		#print("It's not me ("+str(self.num)+").")
		return -1
	"""	
	def rotate(self,signature):
		return signature[3:]+signature[:3]
	"""	
	def rotate(self, l, y=1):
		return np.roll(l,y)		
		

givenTiles = np.empty( (10), dtype=Tile)
for i in range(10):
	givenTiles[i] = Tile.specific(i,baseSignatures[i],numRotations[i],bitmaps[i],False,0)
givenTiles[6].needsExtraProbe = True
givenTiles[6].extraProbeValue = 1.0





window_original_name = '0 Original'
filename = './../Kilian_examples/hrt_top.jpg'
img = cv2.imread(filename)






cv2.namedWindow(window_original_name, cv2.WINDOW_NORMAL)



### Pre-cropping (optional)

#img = img[y0:y1, x0:x1]

### Show original image (RGB)

cv2.imshow(window_original_name,img)
key = cv2.waitKey(0)

### Determine image size

h,w,c = img.shape

### Convert to HSV to create a mask

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
white = np.zeros([h,w,1],dtype=np.uint8)
white.fill(255)
kernel1 = np.ones((4,4),np.uint8)
kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((4,4),np.uint8)

### Create a mask for white and another one for black pixels and remove the noise

mask_white = cv2.inRange(hsv, (0, 0, 189), (153, 14, 255))
mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 90, 66))	

mask_white = cv2.dilate(mask_white,kernel1,iterations = 1)
mask_white = cv2.erode(mask_white,kernel1,iterations = 1)
mask_black = cv2.erode(mask_black,kernel1,iterations = 1)
mask_black = cv2.dilate(mask_black,kernel1,iterations = 1)

### Combine the two masks into one and remove the noise

mask_crop = mask_black+mask_white
mask_crop = mask_crop.clip(0,255).astype("uint8")
mask_crop = cv2.erode(mask_crop,kernel2,iterations = 1)
mask_crop = cv2.dilate(mask_crop,kernel2,iterations = 1)
mask_crop = cv2.dilate(mask_crop,kernel3,iterations = 1)
mask_crop = cv2.erode(mask_crop,kernel3,iterations = 1)

### Detect the full square

_,thresh = cv2.threshold(mask_crop,254,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours: "+str(len(contours)))

if len(contours) != 0:
	cnt = max(contours, key = cv2.contourArea)
	x,y,rw,rh = cv2.boundingRect(cnt)
	#print(x,y,rw,rh)

	mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)

	### Crop all relevant images to detected size

	img_cropped = img[y:y+rh, x:x+rw]
	mask_white = mask_white[y:y+rh, x:x+rw]
	mask_black = mask_black[y:y+rh, x:x+rw]

	
	
	### Draw grid (for preview purposes only)
	
	mask_white_bgr = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
	gridsize = np.array([(rw/5),(rh/5)], dtype=np.uint16)
		
	for col in range(6):
		start_x = col*gridsize[1]
		start_y = 0
		end_x = start_x
		end_y = rh
		cv2.line(mask_white_bgr, (start_x, start_y), (end_x, end_y), (255, 255, 0), 3, 1)		
	for row in range(6):
		start_x = 0
		start_y = row*gridsize[0]
		end_x = rw
		end_y = start_y
		cv2.line(mask_white_bgr, (start_x, start_y), (end_x, end_y), (255, 255, 0), 3, 1)
	
	### Create 25 tiles and probe each of them in 4 places, then save results in signatures[5][5][4]
	
	# first dimension (5) : columns
	# second dimension (5): rows
	# third dimension (4) : probed results: 0: left, 1: right, 2: top, 3: bottom 
	averages = np.zeros(125).reshape(5,5,5)
	signatures = np.zeros(125).reshape(5,5,5)
	identifiers = np.zeros(50).reshape(5,5,2)
	
	rectHalfLong = math.floor(gridsize[0] * 0.5 * 0.5)
	rectHalfShort = 10
	
	identified = 0
	for col in range(5):
		for row in range(5):
			start_x = col*gridsize[0]
			start_y = row*gridsize[1]
			end_x = start_x + gridsize[0]
			end_y = start_y + gridsize[1]
			
			tile = mask_white[start_y:end_y, start_x:end_x]
			tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
			
			sums = np.zeros(5)
			avgs = np.zeros(5)
			
			for c in range(2*rectHalfShort):
				for r in range(2*rectHalfLong):
					val1 = tile[round(gridsize[1]/2-rectHalfLong)+r][round(gridsize[0]/4-rectHalfShort)+c]
					val3 = tile[round(gridsize[1]/2-rectHalfLong)+r][round(3*gridsize[0]/4-rectHalfShort)+c]
					sums[0] += val1
					sums[2] += val3
					
			for c in range(2*rectHalfLong):
				for r in range(2*rectHalfShort):					
					val2 = tile[round(gridsize[1]/4-rectHalfShort)+r][round(gridsize[0]/2-rectHalfLong)+c]
					val4 = tile[round(3*gridsize[1]/4-rectHalfShort)+r][round(gridsize[0]/2-rectHalfLong)+c]			
					sums[1] += val2
					sums[3] += val4
					
			for c in range(2*rectHalfShort):
				for r in range(2*rectHalfShort):
					val5 = tile[round(gridsize[1]/4-rectHalfShort)+r][round(gridsize[0]/4-rectHalfShort)+c]
					sums[4] += val5
			
			avgs = sums / (4*rectHalfLong*rectHalfShort)
			
			for i in range(5):
				averages[col][row][i] = avgs[i]
				if avgs[i] <= 100:
					signatures[col][row][i] = 0
				elif avgs[i] > 100 and avgs[i] <= 170:
					signatures[col][row][i] = 0.5
				elif avgs[i] > 170:
					signatures[col][row][i] = 1
					
			### Identify tile type and orientation
			
			print("\nIdentifying tile (col,row): "+str(col)+","+str(row))			
			print("  with signature: "+str(signatures[col][row]))			
			for givenTile in givenTiles:
				rot = givenTile.isMe(signatures[col][row])
				if rot > -1:
					identified += 1
					print("Found! Tile = "+str(givenTile.num)+", orientation = "+str(rot))
					identifiers[col][row][0] = givenTile.num
					identifiers[col][row][1] = rot
					break
					
	print("\nIdentified tiles: "+str(identified))
			
			
	### Assemble the machine-made equivalent image
	
			
	
	
	
			
			
			
	
	
	
	
	
else:
	print("No contour found.")
	cv2.destroyAllWindows()
	exit(2)

key = cv2.waitKey(0)
cv2.destroyAllWindows()




