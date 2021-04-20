#!/usr/bin/python3
import sys
import cv2
import numpy as np
import math

#np.set_printoptions(threshold=sys.maxsize)



y0 = 1400
y1 = 3800
x0 = 0
x1 = 2592

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = '0 Original'
window_mask_test_name = '1 Mask test'
window_mask_white_name = '2 Mask white'
window_mask_black_name = '3 Mask black'
window_mask_crop_name = '4 Mask full crop'
window_img_cropped_name = '5 Image cropped'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

filename = './../Kilian_examples/hrt_top.jpg'
img = cv2.imread(filename)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_capture_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_capture_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_capture_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_capture_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_capture_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_capture_name, high_V)

cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_mask_test_name, cv2.WINDOW_NORMAL)

cv2.createTrackbar(low_H_name, window_capture_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_capture_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_capture_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_capture_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_capture_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_capture_name , high_V, max_value, on_high_V_thresh_trackbar)

### Pre-cropping (optional)

#img = img[y0:y1, x0:x1]

### Determine image size

h,w,c = img.shape

### Convert to HSV to create a mask

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
white = np.zeros([h,w,1],dtype=np.uint8)
white.fill(255)
kernel1 = np.ones((4,4),np.uint8)
kernel2 = np.ones((2,2),np.uint8)
kernel3 = np.ones((4,4),np.uint8)



while True:
	
	cv2.imshow(window_capture_name,hsv)	
		
	#cv2.imwrite('./../Kilian_examples/hrt_top_hsv.jpg',hsv)
	mask_test = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
	cv2.imshow(window_mask_test_name,mask_test)
	
	key = cv2.waitKey(30)
	if key == ord('q') or key == 27:
		break
	

cv2.namedWindow(window_mask_black_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_mask_white_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_mask_crop_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_img_cropped_name, cv2.WINDOW_NORMAL)

### Create a mask for white and another one for black pixels and remove the noise

mask_white = cv2.inRange(hsv, (0, 0, 189), (153, 14, 255))
mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 90, 66))	

mask_white = cv2.dilate(mask_white,kernel1,iterations = 1)
mask_white = cv2.erode(mask_white,kernel1,iterations = 1)
mask_black = cv2.erode(mask_black,kernel1,iterations = 1)
mask_black = cv2.dilate(mask_black,kernel1,iterations = 1)

cv2.imshow(window_mask_black_name,mask_black)
cv2.imshow(window_mask_white_name,mask_white)	

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
	print(x,y,rw,rh)

	mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
	cv2.rectangle(mask_crop,(x,y),(x+rw,y+rh),(0,255,0),2)

	cv2.imshow(window_mask_crop_name,mask_crop)

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
	cv2.imshow(window_img_cropped_name,mask_white_bgr)	
	
	### Create 25 tiles and probe each of them in 4 places, then save results in signatures[5][5][4]
	
	# first dimension (5) : columns
	# second dimension (5): rows
	# third dimension (4) : probed results: 0: left, 1: right, 2: top, 3: bottom 
	averages = np.zeros(100).reshape(5,5,4)
	signatures = np.zeros(100).reshape(5,5,4)
	
	rectHalfLong = math.floor(gridsize[0] * 0.5 * 0.5)
	rectHalfShort = 10
	
	for col in range(5):
		for row in range(5):
			start_x = col*gridsize[0]
			start_y = row*gridsize[1]
			end_x = start_x + gridsize[0]
			end_y = start_y + gridsize[1]
			
			tile = mask_white[start_y:end_y, start_x:end_x]
			tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
			#cv2.rectangle(mask_crop,(x,y),(x+rw,y+rh),(0,255,0),2)
			cv2.rectangle(tile_bgr,(round(gridsize[0]/4-rectHalfShort),round(gridsize[1]/2-rectHalfLong)),(round(gridsize[0]/4+rectHalfShort),round(gridsize[1]/2+rectHalfLong)),(0,255,0),2)
			cv2.rectangle(tile_bgr,(round(3*gridsize[0]/4-rectHalfShort),round(gridsize[1]/2-rectHalfLong)),(round(3*gridsize[0]/4+rectHalfShort),round(gridsize[1]/2+rectHalfLong)),(0,255,0),2)
			cv2.rectangle(tile_bgr,(round(gridsize[0]/2-rectHalfLong),round(gridsize[1]/4-rectHalfShort)),(round(gridsize[0]/2+rectHalfLong),round(gridsize[1]/4+rectHalfShort)),(0,255,0),2)
			cv2.rectangle(tile_bgr,(round(gridsize[0]/2-rectHalfLong),round(3*gridsize[1]/4-rectHalfShort)),(round(gridsize[0]/2+rectHalfLong),round(3*gridsize[1]/4+rectHalfShort)),(0,255,0),2)
			
			sums = np.zeros(4)
			avgs = np.zeros(4)
			
			for c in range(2*rectHalfShort):
				for r in range(2*rectHalfLong):
					val1 = tile[round(gridsize[1]/2-rectHalfLong)+r][round(gridsize[0]/4-rectHalfShort)+c]
					val2 = tile[round(gridsize[1]/2-rectHalfLong)+r][round(3*gridsize[0]/4-rectHalfShort)+c]
					sums[0] += val1
					sums[1] += val2
					
			for c in range(2*rectHalfLong):
				for r in range(2*rectHalfShort):					
					val3 = tile[round(gridsize[1]/4-rectHalfShort)+r][round(gridsize[0]/2-rectHalfLong)+c]
					val4 = tile[round(3*gridsize[1]/4-rectHalfShort)+r][round(gridsize[0]/2-rectHalfLong)+c]			
					sums[2] += val3
					sums[3] += val4
			
			avgs = sums / (4*rectHalfLong*rectHalfShort)
			
			for i in range(4):
				averages[col][row][i] = avgs[i]
				if avgs[i] <= 100:
					signatures[col][row][i] = 0
				elif avgs[i] > 100 and avgs[i] <= 170:
					signatures[col][row][i] = 0.5
				elif avgs[i] > 170:
					signatures[col][row][i] = 1
			print(signatures[col][row])
			
			#print("Tile" + str(col)+","+str(row)+" (w,h): "+str(tile.shape[1])+","+str(tile.shape[0]))
			cv2.imshow("Tile " + str(col)+","+str(row),tile_bgr)
			key = cv2.waitKey(0)
			cv2.destroyWindow("Tile " + str(col)+","+str(row))
			
			
			
			
			
	
	
	
	
	
else:
	print("No contour found.")
	cv2.destroyAllWindows()
	exit(2)

key = cv2.waitKey(0)
cv2.destroyAllWindows()
