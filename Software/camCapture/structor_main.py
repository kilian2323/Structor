#!/usr/bin/python3
import threading
import cv2
import numpy as np
import random
import structor_cam_v11
import time

win_ui = "Structor GUI"
gui_image = np.zeros((1080,1920,3), np.uint8)

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def gui():
	print("Creating guide window")
	cv2.namedWindow(win_ui, cv2.WND_PROP_FULLSCREEN)
	cv2.moveWindow(win_ui, 0, 0)
	cv2.setWindowProperty(win_ui, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)	
	
	time.sleep(10)
	
	print("Starting structor")
	worker = threading.Thread(target=run_thread_identify, args=("structor::run()",))
	worker.start()
	
	text_style=cv2.FONT_ITALIC
	while(structor_cam_v11.state < 15):	
		gui_image[:,0:1920//2] = random_color() 
		cv2.putText(gui_image,text=str(structor_cam_v11.state),org=(10,200),fontFace=text_style,fontScale=2,color=(0,0,255),thickness=3,lineType=cv2.LINE_AA)
		cv2.imshow(win_ui, gui_image)
		cv2.waitKey(500)


def run_thread_identify(name):
	print("Thread %s: starting", name)
	structor_cam_v11.run()
	print("Thread %s: finishing", name)
    

    


if __name__ == "__main__":
	
	
	
	gui()
	
	
	
