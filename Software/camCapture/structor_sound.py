import pygame

def init_sound():
	pygame.init()
	pygame.mixer.init(48000, 16, 2, 4096)

def play_button():
	while pygame.mixer.music.get_busy():
		continue
	pygame.mixer.music.load("buttonpush.mp3")
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy():
		pygame.time.Clock().tick(1000)
		

def play_timeout():
	while pygame.mixer.music.get_busy():
		continue
	pygame.mixer.music.load("timeout.wav")
	pygame.mixer.music.play()
	while pygame.mixer.music.get_busy():
		pygame.time.Clock().tick(1000)
