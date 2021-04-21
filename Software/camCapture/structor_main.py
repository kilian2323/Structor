#!/usr/bin/python3
import threading
import cv2
import numpy as np
import random
import structor_cam_v11 as structor
import time

win_ui = "Structor GUI"
displaySize = np.array([1920,1080],dtype=int)



###########

gui_state = 0
gui_image = np.zeros((displaySize[1],displaySize[0],3), np.uint8)

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
	
	### Displaying Welcome screen
	gui_state = 1
	print("welcome")	
	
	
	print("Starting structor")
	worker = threading.Thread(target=run_thread_structor, args=("structor::run()",))
	worker.start()
	
	text_style=cv2.FONT_ITALIC
	while(structor.state < 15):	
		gui_image[:,0:1920//2] = random_color() 
		cv2.putText(gui_image,text=str(structor.state),org=(10,200),fontFace=text_style,fontScale=2,color=(0,0,255),thickness=3,lineType=cv2.LINE_AA)
		cv2.imshow(win_ui, gui_image)
		cv2.waitKey(100) # update ten times per second


def run_thread_structor(name):
	print("Thread %s: starting", name)
	structor.run()
	print("Thread %s: finishing", name)
    

    


if __name__ == "__main__":
	
	
	
	gui()
	
	
	
