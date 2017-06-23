import numpy as np 
from PIL import ImageGrab
import cv2
import time

def process_img(original_image):
	# 'Dumbed down' image
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

last_time = time.time()
while(True):
	# Select what part of the screen to capture
	screen = np.array(ImageGrab.grab(bbox=(0,40, 800, 640)))
	print('Loop took {} seconds'.format(time.time()-last_time))
	last_time = time.time()
	cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR.BGR2RGB))
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break