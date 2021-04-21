#!/usr/bin/python3

from __future__ import print_function
import threading
import cv2
import numpy as np
import random
import structor_cam_v11 as structor
import structor_gpio as gpio
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

win_ui = "Structor GUI"
displaySize = np.array([1920,1080],dtype=int)
time1 = 300  # time (seconds) to finish the puzzle
time2 = 15   # additional time (seconds) to finish the puzzle
waitBeforeCapture = 5  # time (seconds) to wait for the user to remove hands
waitAfterAnalysis = 2  # time (seoncds) to wait when analysis reaches 100%
waitAfterFinished = 15 # time (seconds) to display final result screen before restarting the app


###########

gui_state = 0
gui_image = np.empty((displaySize[1],displaySize[0]), np.uint8)
tileSize = 0

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def gui():
	### Creating GUI window
	print("Creating GUI window")
	cv2.namedWindow(win_ui, cv2.WND_PROP_FULLSCREEN)
	cv2.moveWindow(win_ui, 0, 0)
	cv2.setWindowProperty(win_ui, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	
	### Setting up Structor
	print("Setting up Structor")
	structor.setup()
	gpio.start_button_listener()
	
	######## From here: endless loop; application stays open and repeats forever ########
	
	
	while(True):
		gpio.startPressed = False
	
		### Preparing Structor
		gui_state = 1
		print("Initializing Structor objects")
		structor.prepare()	
		
		### Preparing Welcome screen
		gui_state = 2
		print("Creating GUI screen objects")
		gui_bg = create_img_bg()       # create random background image
		gui_text1 = create_img_text("Welcome to Structor.",1000,80,50)      # create texts
		gui_text2 = create_img_text("Push the button to start.",1000,80,50)   
		#cv2.imshow(win_ui, gui_text1)
		#key = cv2.waitKey(0)
		#cv2.imshow(win_ui, gui_text2)
		#key = cv2.waitKey(0)
		
		### Displaying Welcome screen
		gui_state = 2	
		print("Welcome")
		gui_image = gui_bg.copy()
		gui_image = addImages(gui_image,gui_text1,300,300)
		gui_image = addImages(gui_image,gui_text2,300,600)
		
		### TODO: Waiting for user start signal
		gui_state = 3
		print("Waiting for user start signal")
		gpio.startPressed = False
		while gpio.startPressed == False:
			cv2.imshow(win_ui, gui_image)
			cv2.waitKey(1)
			time.sleep(0.01)
		gpio.startPressed = False
			
		### Running countdown timer, waiting for key push or timeout
		gui_state = 4
		timecount = time1
		gui_image = gui_bg.copy()
		gui_text1 = create_img_text("Please assemble your image now.",1000,80,50)    
		gui_text2 = create_img_text("Push the button when you're done.",1000,80,50)  
		gui_timerLine1 = create_img_text("You have",300,80,50)
		gui_timerLine2 = create_img_text("left to finish.",600,80,50)
		gui_image = addImages(gui_image,gui_text1,300,200)
		gui_image = addImages(gui_image,gui_text2,300,300)
		gui_image = addImages(gui_image,gui_timerLine1,300,550)
		gui_image = addImages(gui_image,gui_timerLine2,300,750)
		gpio.startPressed = False
		while timecount >= 0 and gpio.startPressed == False:		
			minutes = int(np.floor(timecount / 60))
			seconds = timecount - minutes * 60
			timerText = "0%d : %02d" % (minutes, seconds)
			gui_timerText = create_img_text(timerText,1000,110,100)
			gui_image = clearOut(gui_image,300,625,1000,110,255)
			gui_image = addImages(gui_image,gui_timerText,300,625)
			time.sleep(0.999)
			cv2.imshow(win_ui, gui_image) 
			cv2.waitKey(1)
			timecount -= 1
		gpio.startPressed = False
			
		if timecount < 0:
			timecount = 0
			### "Hurry up" timer (additional 15 seconds maximum)
			gui_image = gui_bg.copy()
			gui_timerLine1 = create_img_text("Please finish within 15 seconds!",1000,80,50)
			gui_image = addImages(gui_image,gui_timerLine1,300,200)
			gpio.startPressed = False
			while timecount <= time2 and gpio.startPressed == False:
				minutes = int(np.floor(timecount / 60))
				seconds = timecount - minutes * 60
				timerText = "+ 0%d : %02d" % (minutes, seconds)
				gui_timerText = create_img_text(timerText,1000,160,100)
				gui_image = clearOut(gui_image,300,500,1000,160,255)
				gui_image = addImages(gui_image,gui_timerText,300,500)
				time.sleep(0.999)
				cv2.imshow(win_ui, gui_image) 
				key = cv2.waitKey(1)
				timecount += 1
			gpio.startPressed = False
				
		### The image is finished now	
		gui_state = 5
		gui_image = gui_bg.copy()
		gui_text1 = create_img_text("We are taking a picture of your work.",1500,80,50)
		gui_text2 = create_img_text("Please remove your hands from the table.",1500,80,50)
		gui_image = addImages(gui_image,gui_text1,300,300)
		gui_image = addImages(gui_image,gui_text2,300,600)
		cv2.imshow(win_ui, gui_image) 
		cv2.waitKey(1)
		time.sleep(waitBeforeCapture)
		
		### Starting structor capture
		print("Starting structor capture")
		gui_state = 6
		structor.capture()
		
		### Picture captured, now let's identify in a parallel thread
		print("Starting structor identify")
		gui_state = 7
		worker = threading.Thread(target=run_thread_structor, args=("structor::identify()",))
		worker.start()
		
		gui_state = 8
		gui_image = gui_bg.copy()
		gui_text1 = create_img_text("Identifying...",1000,80,50)
		gui_image = addImages(gui_image,gui_text1,300,300)
		pct = 0.0
		while structor.state <= 15:
			pct_max = 100.0 * (structor.state - 6) / 9.0
			pct += 1.5
			pct = min(pct,pct_max)
			gui_image = clearOut(gui_image,300,800,1600,tileSize,255)
			#gui_pct_text = create_img_text(str(pct)+"%",100,80,25)		
			gui_image = cv2.rectangle(gui_image, (300,800),(300+int(pct/100.0*1300),800+tileSize),0,-1)
			gui_image = cv2.rectangle(gui_image, (300,800),(1600,800+tileSize),0,2)
			#gui_image = addImagesTransparent(gui_image,gui_pct_text,350,850,255)
			cv2.imshow(win_ui, gui_image) 
			key = cv2.waitKey(5)
			if structor.state == 15:
				if pct >= 100.0:
					pct = 100.0				
					break
		time.sleep(waitAfterAnalysis)
			
		# Identification finished
		print("Identification finished")
		gui_state = 9
		gui_image = gui_bg.copy()
		if structor.resultObject.detection_success == True:
			gui_text1 = create_img_text("Successfully detected.",600,80,50)
		else:
			gui_text1 = create_img_text("Could not detect any valid pattern.",1200,80,50)	
		gui_text2 = create_img_text("Your signature:",550,80,50)	
		
		string_signature = ""
		#for i in range(len(structor.resultObject.result_signature.flatten())):
		#	string_signature += (str(structor.resultObject.result_signature.flatten()[i]) + " ")
		for row in range(5):
			for col in range(5):
				string_signature += (str(structor.resultObject.result_signature[col][row][0]) + " " + str(structor.resultObject.result_signature[col][row][1]) + "  ")
			string_signature += "\n\n"
		gui_text_signature = create_img_text(string_signature,850,550,50)
		resultImage = cv2.resize(structor.resultObject.img_result,(500,500))		
		
		gui_image = addImages(gui_image,gui_text1,300,200)
		gui_image = addImages(gui_image,gui_text2,830,400)
		gui_image = addImages(gui_image,gui_text_signature,830,500)
		gui_image = addImages(gui_image,resultImage,300,400)
		cv2.rectangle(gui_image, (295,395),(295+510,395+510),0,2)
		
		timeCounter = 0
		gpio.startPressed = False
		while gpio.startPressed == False:		
			cv2.imshow(win_ui, gui_image) 
			cv2.waitKey(1)
			time.sleep(0.999)
			timeCounter += 1
			if timeCounter > waitAfterFinished:
				break
		gpio.startPressed = False
		
	


def run_thread_structor(name):
	print("----- Thread %s: starting" % name)
	structor.identify()
	print("----- Thread %s: finishing" % name)

    

def create_img_bg():
	global tileSize
	image = np.empty((displaySize[1],displaySize[0]), np.uint8)
	image.fill(255)
	num_tiles = 10
	tileSize = int(round(displaySize[1] / num_tiles))
	for t in range(num_tiles):
		tileNum = random.randint(0,9)
		thisTileObj = structor.givenTiles[tileNum]
		tileRot = random.randint(0,thisTileObj.numRotations)
		posY = t * tileSize
		thisTile = thisTileObj.rotImage(tileRot)
		thisTile = cv2.resize(thisTile, (tileSize,tileSize))		
		image[posY:posY+tileSize,0:tileSize] = thisTile  
	return image
	
def create_img_text(_text,w,h,size):
	#font = ImageFont.truetype("gravtrac compressed bd.ttf", size, encoding="unic")
	font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size, encoding="unic")
	# create a white 8-bit grayscale canvas
	canvas = Image.new('L', (w, h), color=255)
	# draw the text onto the text canvas, and use black as the text color
	draw = ImageDraw.Draw(canvas)
	draw.text((5,5), _text, 0, font)
	return np.array(canvas)
	
def clearOut(img,x,y,w,h,color):
	image = img.copy()
	image[y:y+h,x:x+w] = color
	return image
	
	
def addImages(img1, img2, x, y):
	image = img1.copy()
	shape1 = np.shape(img1)
	shape2 = np.shape(img2)
	endX = min(shape1[1],x+shape2[1])
	endY = min(shape1[0],y+shape2[0])
	image[y:endY,x:endX] = img2.copy()
	return image
	
def addImagesTransparent(img1, img2, x, y, trColor):
	image = img1.copy()
	shape1 = np.shape(img1)
	shape2 = np.shape(img2)
	endX = min(shape1[1],x+shape2[1])
	endY = min(shape1[0],y+shape2[0])	
	image_cropped = image[y:endY,x:endX]
	for row in range(endY-y):
		for col in range(endX-x):
			if img2[row][col] != trColor:
				image_cropped[row][col] = img2[row][col]	
	#cv2.imshow("new",image_cropped)
	#cv2.waitKey(0)
	image = addImages(image,image_cropped,x,y)
	
	return image
			
			
	


if __name__ == "__main__":
	
	
	
	gui()
	
	
	
