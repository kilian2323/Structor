#!/usr/bin/python3

import RPi.GPIO as GPIO
import time
import os
import structor_main as main
  

btn_start_pin = 7 # board pin 7
btn_exit_pin = 11
debounce = 500 # milliseconds between falling edges



startPressed = False # the flag for the button listener
start_lastPush = 0

def btn_start_falling_event(btn_start_pin):
	global startPressed
	global start_lastPush	
	if millis() - start_lastPush > debounce:
		print("Start button pushed")
		startPressed = True 
	start_lastPush = millis()
	
def btn_exit_falling_event(btn_start_pin):
	main.cv2.destroyAllWindows()
	print("Clean exit")
	os._exit(1)
	
def init_gpio():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(btn_start_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
	GPIO.setup(btn_exit_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
	GPIO.add_event_detect(btn_start_pin, GPIO.FALLING, callback=btn_start_falling_event)
	GPIO.add_event_detect(btn_exit_pin, GPIO.FALLING, callback=btn_exit_falling_event)




def millis():
	return round(time.time() * 1000)
	


