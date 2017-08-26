import numpy as np 
from PIL import ImageGrab
import cv2
import time
from direct_keys import PressKey, ReleaseKey, W, A, S, D
import pyautogui

# Region of Interest
def roi(img, vertices):
	# Create matrix of shape 'img'
	mask = np.zeros_like(img)
	# Fill the mask by the vertices
	cv2.fillPoly(mask, vertices, 255)
	# Leave only region of interest
	masked = cv2.bitwise_and(img, mask)
	return masked

# Draw road lanes
def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
	# If this fails, go with some default line
	try:
		# Finds the maximum y value for a lane marker
		# (Since we cannot assume the horizon will always be at the same point)
		ys = []
		for i in lines:
			for ii in i:
				ys += [ii[1],ii[3]]
		min_y = min(ys)
		max_y = 600
		new_lines = []
		line_dict = {}

		for idx, i in enumerate(lines):
			for xyxy in i:
				# These four lines:
				# Modified from 
				# Used to calculate the definition of a line, given two sets of coords
				x_coords = (xyxy[0],xyxy[2])
				y_coords = (xyxy[1],cycy[3])
				A = vstack([x_coords, ones(len(x_coords))]).T
				m, b = lstsq(A, y_coords)[0]

				# Calculate our new, and improved xs
				x1 = (min_y - b) / m
				x2 = (max_y - b) / m

				line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
				new_lines.append([int(x1), min_y, int(x2), max_y])

		final_lanes = {}
		# Find lines with similar slopes
		for idx in line_dict:
			final_lanes_copy = final_lanes.copy()
			m = line_dict[idx][0]
			b = line_dict[idx][1]
			line = line_dict[idx][2]

			if len(final_lanes) == 0:
				final_lanes[m] = [[m, b, line]]
			else:
				found_copy = False
				for other_ms in final_lanes_copy:
					if not found_copy:
						if abs(other_ms*1.1) > abs(m) > abs(other_ms*0.9):
							if abs(final_lanes_copy[other_ms][0][1]*1.1) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
								final_lanes[other_ms].append([m,b,line])
								found_copy = True
								break
						else:
							final_lanes[m] = [[m, b, line]]
		line_counter = {}
		for lanes in final_lanes:
			line_counter[lanes] = len(final_lanes[lanes])

		top_lanes = sorted(line_counter.items(), key=lambda item:item[1])[::-1]

		lane1_id = top_lanes[0][0]
		lane2_id = top_lanes[1][0]

		# Find the two most common slopes
		def average_lane(lane_data):
			x1s = []
			y1s = []
			x2s = []
			y2s = []
			for data in lane_data:
				x1s.append(data[2][0])
				y1s.append(data[2][1])
				x2s.append(data[2][2])
				y2s.append(data[2][3])
			return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))

		l1_x1, l1_y1, l1_x1, l1_y2 = average_lane(final_lanes[lane1_id])
		l2_x1, l2_y1, l2_x1, l2_y2 = average_lane(final_lanes[lane2_id])

		return [l1_x1, l1_y1, l1_x1, l1_y2], [l2_x1, l2_y1, l2_x1, l2_y2], lane1_id, lane2_id
	except Exception as e:
		print(str(e))

# Draw lines from coordinates
def draw_lines(img, lines):
	try:
		for line in lines:
			coords = line[0]
			# Draw Line on image, no need to return image
			cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 3)
	except:
		pass

# Turn Image to Black/White With only Edges/Corners
def process_img(original_image):
	# 'Dumbed down' image
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	# Edge Detection
	processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
	# Blur image
	processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
	# Edges for Region of Interest
	vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])
	# Image Region of Interest
	processed_img = roi(processed_img, [vertices])
	# Detect Lines: RRes, theta, threshold, minLineLength, maxLineGap
	lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 30, 15)
	# Draw Lines
	draw_lines(processed_img, lines)
	return processed_img, original_image, m1, m2

def straight():
	PressKey(W)
	ReleaseKey(A)
	ReleaseKey(D)

def left():
	PressKey(A)
	ReleaseKey(W)
	ReleaseKey(D)

def right():
	PressKey(D)
	ReleaseKey(A)
	ReleaseKey(W)

def slow_down():
	ReleaseKey(W)
	ReleaseKey(A)
	ReleaseKey(D)

for i in list(range(g))[::-1]:
	print(i + 1)
	time.sleep(1)

def main():
	last_time = time.time()
	while True:
		# Select what part of the screen to capture
		screen = np.array(ImageGrab.grab(bbox=(0,40, 800, 640)))
		new_screen, original_image, m1, m2 = processed_img(screen)
		print('Loop took {} seconds'.format(time.time()-last_time))
		last_time = time.time()
		# cv2.imshow('window', new_screen)
		cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR.BGR2RGB))
		if m1 < 0 and m2 < 0:
			right()
		elif m1 > 0 and m2 > 0:
			left()
		else:
			straight()
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break