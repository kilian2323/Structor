#!/usr/bin/python3

from gpiozero import Button

startButton = Button(4, pull_up = True, bounce_time = 0.4) # GPIO4; pin 7
startPressed = False # the flag for the button listener

def start_pressed():
	global startPressed
	print("Start button was pressed")
	startPressed = True

def start_button_listener():
	startButton.when_pressed = start_pressed
		



	
