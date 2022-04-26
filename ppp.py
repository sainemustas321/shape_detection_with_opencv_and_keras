from importlib.resources import path
from tkinter import NONE
import numpy as np
import cv2, time
import matplotlib.pyplot as plt
import numpy as np
import glob
from keras.models import load_model
import pytesseract
import math
from math import acos, degrees, atan
import geopy
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import pytesseract
import colorsys
from skimage import data
import webcolors
from collections import Counter
import random
from torch import le


pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
shapepath = glob.glob('D:/work/avengers_assemble/saved_pic/*.jpg')

#from kmeans_pytorch import kmeans

model = load_model('./model/keras_model7.h5')
#classes = ['triangle', 'trapezoid', 'star', 'square', 'semicircle', 'quatercircle', 'plus', 'pentagon', 'circle']
classes = ['CIRCLE', 'PENTAGON', 'CROSS', 'QUATER_CIRCLE', 'SEMICIRCLE', 'SQUARE', 'STAR', 'TRAPEZOID', 'TRIANGLE']
correct = []

print(len(correct))
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


light_green = (9,0,0)
dark_green = (71,255,235)

light_brown = (10,0,0)
dark_brown = (30,255,219)

light_ground = (16,66,0)#yellow line
dark_ground = (26,154,255)

light_road = (0,0,0)
dark_road = (255,91,221)

BLACK = ["black"]
GRAY = ["dimgray", "dimgrey", "gray", "grey", "darkgray", "darkgrey", "silver", "lightgray", "lightgrey", "gainsboro"]
WHITE = ["whitesmoke", "white", "snow", "linen", "antiquewhite", "oldlace", "floralwhite", "ivory", "mintcream",
"azure", "aliceblue", "ghostwhite", "lavenderblush"]

RED = ["rosybrown", "lightcoral", "indianred", "brown", "firebrick","maroon", "darkred", "red", "mistyrose", "salmon"
, "deeppink", "hotpink", "palevioletred", "crimson", "pink", "lightpink"]
ORANGE = ["tomato", "darksalmon", "coral", "orangered", "lightsalmon", "peachpuff", "bisque", "darkorange", "orange",
"navajowhite","blanchedalmon", "papayawhip","moccasin", "wheat", "darkgoldenrod", "golldenrod"]

BROWN = ["sienna", "chocolate", "saddlebrown", "sandybrown", "peru", "burlywood", "tan"]
YELLOW = ["gold", "yellow", "cornsilk", "lemonchiffon", "khaki", "palegoldenrod", "darkkhaki", "beige", "lightyellow", "lightgoldenrodyellow"]
GREEN = ["olive","olivedrab", "yellowgreen", "darkolivegreen", "greenyellow", "chartreuse", "lawngreen", "honeydew", "darkseagreen",
"palegreen", "lightgreen", "forestgreen", "limegreen", "green", "darkgreen", "lime", "seagreen", "mediumseagreen","springgreen", "mediumspringgreen", 
"mediumaquamarine", "aquamarine", "turquoise", "lightseagreen", "mediumsturquoise", "darkslategray", "darkslategrey","teal", "darkcyan"]

BLUE = ["lightcyan", "paleturquoise", "aqua", "cyan", "darkturquoise", "cabetblue", "powderblue", "lightblue", "deepskyblue",
"skyblue", "lightskyblue", "steelblue","dodgerblue", "lightslategray", "lightslategrey", "slategray","slategrey", "lightsteelblue"
, "cornflowerblue", "royalblue", "midnightblue", "navy", "darkblue", "mediumblue", "blue"]

PURPLE = ["lavender", "slateblue", "darkslateblue", "mediumslateblue", "mediumpurple", "rebeccapurple", "blueviolet", "indigo", "darkorchid", 
"darkviolet", "mediumorchid", "thistle", "plum", "violet", "purple", "darkmagenta", "fuchsia", "magenta", "orchid", "mediumvioletred"]

shapename = []
name_file= []
shape_file = []
pos = []
check_positions = []

def bg_cut(path,count_crop):
	
	#stt = time.time()
	#path = glob.glob(path)
	print(path)
	#for filename in path:
		
	img = cv2.imread(path)
	#img = cv2.bilateralFilter(img,9,75,75)
	#kernel = np.ones((5,5), np.uint8)
	#img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
	st2 = time.time()
	Z = np.float32(img.reshape((-1,3)))
	#print(Z.shape)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
	K = 16
	_,labels,centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
	#print(centers)
	labels = labels.reshape((img.shape[:-1]))
	#print('labelsss 2:',labels)

	pos = []
	#result   = [np.hstack([img, reduced])]
	#print('Time per loop: ',time.time()- st2)
	for i, c in enumerate(centers):

		mask = cv2.inRange(labels, i, i)

		mask = cv2.GaussianBlur(mask ,(5,5),0)

		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.imwrite(f'./testdata/{count_crop}_{str(i).rjust(4, "0")}.jpg',mask)
		
		areas = []
		
		for cnt in contours:
				
			area = cv2.contourArea(cnt)
			areas.append(area)
			#(x, y, w, h) = cv2.boundingRect(cnt)
			#cv2.rectangle(mask, (x, y), (x + w, y + h), (255,0,0), 5)
			#print(area)
			
		aaa =  sum(areas)
		#print(aaa)
		aa1 = []
		if 500 < aaa < 15000:
			countt = 0
			for cnt in contours:
				
				area = cv2.contourArea(cnt)
				#print(area)
				aa1.append(area)
				#c = max(contours, key = cv2.contourArea)
				#print(c)
				#print(aa1)
				if area >= 500 and area <= max(aa1):
					(x, y, w, h) = cv2.boundingRect(cnt)
					cv2.rectangle(mask, (x, y), (x + w, y + h), 255, 0, 0, 5)
					#cv2.rectangle(img, (x, y, w, h), (255,0,0), 2)
					fit = img[y+10:y -20 + h, x+10:x -20 + w]
					x = int(x-50)
					y = int(y-50)
					if x < 0 :
						x = 0
					if y < 0 :
						y = 0

					roi = img[y:y +100 + h, x:x + 100 + w]
					mask_roi = mask[y:y +100 + h, x:x + 100 + w]
					
					print("area ; ",area)
					#roi = cv2.resize(roi, (640,480),interpolation=cv2.INTER_AREA)
					center_x = int((x+x+w)/2 )
					center_y = int((y+y + h)/2)
					
					pos_ob = (center_x, center_y)
					print(pos_ob, f'saved_pic/{count_crop}_{countt}.jpg')
					pos.append(pos_ob)
					#print('Time: ', time.time()-stt)
					#plt.imshow(roi)
					#plt.show()
					#cv2.imwrite('1.jpg',roi)
					#print(f'saved_pic/{path}_{i}.jpg')
					try:
						cv2.imwrite(f'./saved_pic/{count_crop}_{str(countt).rjust(4, "0")}.jpg',roi)
						#cv2.imwrite(f'./fitcrop/{count_crop}_{str(i).rjust(4, "0")}.jpg',fit)
						cv2.imwrite(f'./mask_roi/{count_crop}_{str(countt).rjust(4, "0")}.jpg',mask_roi)
						countt += 1
						#cv2.imwrite(f'mask_roi/{count_crop}_{i}.jpg',mask_roi)
					except Exception as e:
						print(e)

						#print(roi, x, y, w, h)
					
					print("-------------Done Cropping-------------")
	#print('position of object : ', pos)1
	print(pos)
	return pos

def shape_name(shapepath,shapepath2):
	shapename = "Undefined"
	print('---------------------shape check---------------------')
	
	

	name_file.append(shapepath)
	image = cv2.imread(shapepath)
	image2 = cv2.imread(shapepath2)
	sss = shapepath.split('/')[-1].split('.')[0]
		
	size = (224, 224)
	#print("qpwodkpqwdkpqwkdopqwkdp[kqwopdqwd")
		#image = ImageOps.fit(image, size, Image.ANTIALIAS)
	image = cv2.resize(image, size)
			#turn the image into a numpy array
	#blurred = cv2.bilateralFilter(image,9,75,75)
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	

	_, img_thresh = cv2.threshold(hsv, 150, 255, cv2.THRESH_BINARY_INV)#
	contours, _  = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	img_thresh = cv2.resize(img_thresh, size)	
	img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB)
		
	image_array = np.asarray(image)
		# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
		# Load the image into the array
	data[0] = normalized_image_array
	prediction = model.predict(data)
	class2s = np.argmax(prediction)
	max_val = np.amax(prediction)
	add = 0	
	for cnt in contours:
	
		area = cv2.contourArea(cnt)
		c = max(contours, key = cv2.contourArea)
		p_val, th = .4, .5
		#print(area)
			#cv2.drawContours(image, contours, 3, (0,255,0), 3)
			#print(area)
		#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
		(x, y, w, h) = cv2.boundingRect(c)
		
		if area >= 1500 and area <= 10000:
			print(max_val)
			#cv2.imwrite(f'D:/work/avengers_assemble/mask_roi/{sss}_{classes[class2s]}_{add}.jpg',img_thresh)
			if  max_val > p_val:
				#cv2.rectangle(image, (x, y, w, h), (255,0,0), 2)
				#cv2.putText(image, classes[class2s], (x, w), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,25,255), 1)
				print(classes[class2s])
				shapename = classes[class2s]
				#print(max_val)
				#shapename.append(classes[class2s])
				shape_file.append(shapepath)
				#print()
				fit_tooo = image2[y+62:y-124 + h, x+62:x-124 + w]
				
				try:
					print('-----strating save image-----')
					#print(shapepath)
					cv2.imwrite(f'D:/work/avengers_assemble/shape/{sss}_{classes[class2s]}_{add}.jpg',image2)
				#cv2.imwrite('./shape/{}-{}-shape.jpg'.format(shapepath, classes[class2s]), image)
					cv2.imwrite(f'D:/work/avengers_assemble/ans_auto/{sss}_{classes[class2s]}_{add}.jpg',image2)
					cv2.imwrite(f'D:/work/avengers_assemble/fitcrop/{sss}_{classes[class2s]}_{add}.jpg', fit_tooo)
					
					add+=1
				except Exception as e:
					print(e)
				
				
				#with open('name_shape.txt', 'a') as f:
				#	f.write('Shape name: {}\n'.format(classes[class2s]))
				#with open('name_file.txt', 'a') as f:
				#	f.write('{}\n'.format(image))
				
				print("-------img saved-------")
	#cv2.imshow("img_thresh", img_thresh)
	#cv2.imshow('maskl', mask)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
				#print(i)
#print(name)
#print(name_image)
			#name.append('cant')
		
	
			#print(name)
	return shapename, name_file, shape_file

def getPos(pos, name_file, shape_file):
	print('---------------------positioning---------------------')
	print(pos, name_file, shape_file)
	
	for j in range(len(shape_file)):
				# with open('position.txt', 'a') as f:
				# 	f.write('{}\n'.format(pos[i]))
			check_positions.append(pos[j])
	
			
	return check_positions

def lat_long(lat,long, DRONE_ALTITUDE,uav_angle, pos):
	heading = "Undefined"
	Width = 6960
	Height = 4640
	#check_positions = [0,0]
	
	mid_x = Width/2
	mid_y = Height/2
	#uav_angle = uav_angle

	# แปลง lat >> radian
	lat1 = math.radians(lat)
	long1 = math.radians(long)


	#info ของ กล้อง
	I = Width
	F = 0.708661417 #focal length
	SW = 0.881889764 #sensor width    
	H = DRONE_ALTITUDE



	def meter(Ox,Oy):
		GSD = (SW*H)/(F*I)
		#print(GSD)
		xDistance = Ox*GSD
		yDistance = Oy*GSD
		inches = math.sqrt(((xDistance)-(mid_x))**2+((yDistance)-(mid_y))**2)
		meters = inches*GSD
		meters = meters/39370.0787
		print("meter : ", meters)
		return xDistance, yDistance, meters
	#print(xLength, yLength)

	#for image_position in pos:image_position =
	image_position = pos
	print('\n---------------------find Object lat & long---------------------')
	xDistance, yDistance, meters = meter(image_position[0],image_position[1])
	print('image_positoin : ' , image_position[0],image_position[1])
	#distance 1m -->>> pixel
	#l = math.sqrt(abs((meters**2)-(KNOWN_DISTANCE**2)))
	#print(xDistance, yDistance, meters)
		#หามุมbearing
	brng = degrees(atan(xDistance / yDistance))
	print("Calculated angle = {:.2f} ".format(brng))
		# hdg เครื่องไป 0 brng
	origin = geopy.Point(lat, long)
		
		#print(xDistance,yDistance)
		
		#Q2 brng = 360 - brng
	if 0 < image_position[0] < mid_x and 0 < image_position[1] < mid_y:
		destination = geodesic(meters).destination(origin, 360-brng + uav_angle)
		new_lat, new_long = destination.latitude, destination.longitude
		orientation = uav_angle
		
		if 337.5 <= orientation < 22.5:
			heading = "N"
		elif  22.5<= orientation < 67.5:
			heading = "NE"
		elif  67.5<= orientation < 112.5:
			heading = "E"
		elif  112.5<= orientation < 157.5:
			heading = "SE"
		elif  157.5<= orientation < 202.5:
			heading = "S"
		elif  202.5<= orientation < 247.5:
			heading = "SW"
		elif  247.5<= orientation < 292.5:
			heading = "W"
		elif  292.5<= orientation < 337.5:
			heading = "NW"
		print(f"Orientation Bearing & Orientation = {orientation} {heading}")
		print("latitude and longitude = ",new_lat,new_long)

		print("Q2")
			#print(new_lat,new_long)

		#Q1
	elif mid_x < image_position[0] < Width and 0 < image_position[1] < mid_y:
			#print("Q1")
		destination = geodesic(meters).destination(origin, brng + uav_angle)
		new_lat, new_long = destination.latitude, destination.longitude
		orientation = uav_angle
		if 337.5 <= orientation < 22.5:
			heading = "N"
		elif  22.5<= orientation < 67.5:
			heading = "NE"
		elif  67.5<= orientation < 112.5:
			heading = "E"
		elif  112.5<= orientation < 157.5:
			heading = "SE"
		elif  157.5<= orientation < 202.5:
			heading = "S"
		elif  202.5<= orientation < 247.5:
			heading = "SW"
		elif  247.5<= orientation < 292.5:
			heading = "W"
		elif  292.5<= orientation < 337.5:
			heading = "NW"
		print(f"Orientation Bearing & Orientation = {orientation} {heading}")
		print("latitude and longitude = ",new_lat,new_long)
		print("Q1")
		#Q3 brng = brng + 180
	elif 0 < image_position[0] < mid_x and mid_y < image_position[1] < Height:
		destination = geodesic(meters).destination(origin, 180 + brng + uav_angle)
		new_lat, new_long = destination.latitude, destination.longitude
		orientation = uav_angle
		if 337.5 <= orientation < 22.5:
			heading = "N"
		elif  22.5<= orientation < 67.5:
			heading = "NE"
		elif  67.5<= orientation < 112.5:
			heading = "E"
		elif  112.5<= orientation < 157.5:
			heading = "SE"
		elif  157.5<= orientation < 202.5:
			heading = "S"
		elif  202.5<= orientation < 247.5:
			heading = "SW"
		elif  247.5<= orientation < 292.5:
			heading = "W"
		elif  292.5<= orientation < 337.5:
			heading = "NW"
		print(f"Orientation Bearing & Orientation = {orientation} {heading}")
		print("latitude and longitude = ",new_lat,new_long)
		print("Q3")

		#Q4 brng = 180 - brng
	else:
		destination = geodesic(meters).destination(origin, 180 - brng + uav_angle)
		new_lat, new_long = destination.latitude, destination.longitude
		orientation = uav_angle
		if 337.5 <= orientation < 22.5:
			heading = "N"
		elif  22.5<= orientation < 67.5:
			heading = "NE"
		elif  67.5<= orientation < 112.5:
			heading = "E"
		elif  112.5<= orientation < 157.5:
			heading = "SE"
		elif  157.5<= orientation < 202.5:
			heading = "S"
		elif  202.5<= orientation < 247.5:
			heading = "SW"
		elif  247.5<= orientation < 292.5:
			heading = "W"
		elif  292.5<= orientation < 337.5:
			heading = "NW"
		print(f"Orientation Bearing & Orientation = {orientation} {heading}")
		print("latitude and longitude = ",new_lat,new_long)
		print("Q4")

	with open('geo.txt', 'a') as f:
		f.write('lat : {}\nlong: {}\n'.format(new_lat, new_long))
	return new_lat, new_long, heading

def source(Data_path):

	Data = []

	f = open(Data_path, "r") 
	data_text = f.read().split(',')
	data_text[4]
	return data_text

def RGB2HEX(color):
	return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
	
def find_alpha_color(path):
	start = time.time()
	image = cv2.imread(path)
	#kernel = np.ones((3, 3), np.uint8)
	#image = cv2.filter2D(image, -1, kernel)
	image =  cv2.bilateralFilter(image,9,75,75)
	#image = gammaCorrection(image, 1.5)
	image = cv2.blur(image, (5,5))
	# แปลงimg >> rgb  แล้ว  ไปเป็ร gray_image >>> resize, ใหเ px มันขยาย
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	modified_image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
	
	clf = KMeans(n_clusters = 2) #<<<< จน.สีที่อยากหา
	labels = clf.fit_predict(modified_image)
	
	counts = Counter(labels)
	# sort to ensure correct color percentage
	counts = dict(sorted(counts.items()))
	
	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]
	
	list_rgb = list(counts.values())
	#print(rgb_colors)
	#print(hsv_colors)
	max_value = min(list_rgb)
 
	index_value = list_rgb.index(max_value)
	rgb_pos = rgb_colors[index_value]
	
	r = rgb_pos[0]
	g = rgb_pos[1]
	b = rgb_pos[2]
	#r = colorsys.rgb_to_hsv(r/255,g/255,b/255)[0]
	#g = colorsys.rgb_to_hsv(r/ 255,g/ 255,b/255)[1]
	#b = colorsys.rgb_to_hsv(r/ 255,g/ 255,b/ 255)[2]
	r = int(r)
	g = int(g)
	b = int(b)
	#plt.figure(figsize = (8, 6))
	plt.pie(counts.values(), labels = hex_colors, colors = hex_colors, autopct='%.2f%%')

	end = time.time()
	total = end - start
	print("time :", total)
	#print(int(r*180),int(g*255),int(b*255))
	print("r g b : ", r,g,b) 
	#plt.imshow(image)
	#plt.show()
	requested_alpha_color = (r ,g , b)
	#cv2.imshow('image',modified_image)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	return requested_alpha_color

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv2.LUT(src, table)

def find_shape_color(path):
	start = time.time()
	image = cv2.imread(path)
	#kernel = np.ones((3, 3), np.uint8)
	#image = cv2.filter2D(image, -1, kernel)
	image =  cv2.bilateralFilter(image,9,75,75)
	#image = gammaCorrection(image, 1.5)
	#image = cv2.blur(image, (5,5))
	# แปลงimg >> rgb  แล้ว  ไปเป็ร gray_image >>> resize, ใหเ px มันขยาย
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	modified_image = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
	modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
	
	clf = KMeans(n_clusters = 3) #<<<< จน.สีที่อยากหา
	labels = clf.fit_predict(modified_image)
	
	counts = Counter(labels)
	# sort to ensure correct color percentage
	counts = dict(sorted(counts.items()))
	
	center_colors = clf.cluster_centers_
	# We get ordered colors by iterating through the keys
	ordered_colors = [center_colors[i] for i in counts.keys()]
	hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
	rgb_colors = [ordered_colors[i] for i in counts.keys()]
	
	list_rgb = list(counts.values())
	#print(rgb_colors)
	#print(hsv_colors)
	max_value = max(list_rgb)
 
	index_value = list_rgb.index(max_value)
	rgb_pos = rgb_colors[index_value]
	
	r = rgb_pos[0]
	g = rgb_pos[1]
	b = rgb_pos[2]
	#r = colorsys.rgb_to_hsv(r/255,g/255,b/255)[0]
	#g = colorsys.rgb_to_hsv(r/ 255,g/ 255,b/255)[1]
	#b = colorsys.rgb_to_hsv(r/ 255,g/ 255,b/ 255)[2]
	# r = int(r*180)
	# g = int(g*255)
	# b = int(b*255)
	#plt.figure(figsize = (8, 6))
	r = int(r)
	g = int(g)
	b = int(b)
	plt.pie(counts.values(), labels = hex_colors, colors = hex_colors, autopct='%.2f%%')

	end = time.time()
	total = end - start
	print("time :", total)
	#print(int(r*180),int(g*255),int(b*255))
	print("r g b : ", r,g,b) 
	#plt.imshow(image)
	#plt.show()
	requested_shape_color = (r ,g , b)
	#cv2.imshow('image',modified_image)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	return requested_shape_color

def define_color(requested_color):
	
	name_of_color = "Undefined"
	r = requested_color[0]
	g = requested_color[1]
	b = requested_color[2]
	
	if 0 < r < 255 and 0 < g < 57 and 128 < b < 255:
		name_of_color = 'White'
		print(name_of_color)

	elif 9 < r < 22 and 104 < g < 255 and 164 < b < 255:		
		name_of_color = 'Orange'
		print(name_of_color)

	elif 23 < r < 33 and 20 < g < 255 and 10 < b < 255:		
		name_of_color = 'YELLOW'
		print(name_of_color)

	elif 34 < r < 90 and 10 < g < 255 and 10 < b < 255:		
		name_of_color = 'GREEN'
		print(name_of_color)

	elif 91 < r < 144 and 40 < g < 255 and 18 < b < 255:		
		name_of_color = 'BLUE'
		print(name_of_color)

	elif 145 < r < 169 and 30 < g < 255 and 30 < b < 255:		
		name_of_color = 'PURPLE'
		print(name_of_color)

	elif 0 < r < 180 and 0 < g < 255 and 0 < b < 26:		
		name_of_color = 'GREY'
		print(name_of_color)

	elif 0 < r < 9 and 20 < g < 255 and 10 < b < 255:		
		name_of_color = 'RED'
		print(name_of_color)

	elif 170 < r < 255 and 20 < g < 255 and 10 < b < 255:		
		name_of_color = 'RED2'
		print(name_of_color)

	return name_of_color

def rotate_image(image, angle):
	
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

def text(path):

	txts = []
	conf = []

	# resize image
		
	img = cv2.imread(path)
	# scale_percent = 200 # percent of original size
	# width = int(img.shape[1] * scale_percent / 100)
	# height = int(img.shape[0] * scale_percent / 100)
	# dim = (width, height)
	# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	

	# print('Resized Dimensions : ',resized.shape)
	# img = resized
	#img = cv2.resize(img,(0,0),fx=3,fy=3)
	img = cv2.GaussianBlur(img,(11,11),0)
	img = cv2.resize(img, (300, 300))
	#print(pytesseract.image_to_string(img))
	#img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT,value=[0])
	img = cv2.medianBlur(img,9)

	kernel = np.ones((3, 3), np.uint8)
	img = cv2.erode(img, kernel)
	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	_, img_thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
	
	config= '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ' #   

	for angle in np.arange(0, 450, 90):

		result2 = rotate_image(img_thresh, angle)

		test1 = pytesseract.image_to_string(result2, lang='eng', config=config)
		
		data = pytesseract.image_to_data(result2, output_type='data.frame')
		#print(data)
		#print()
		n_boxes = len(data['text'])
		#print(data['text'])
		for i in range(n_boxes):
			
			#print(int(data['conf'][i]))
			if int(data['conf'][i]) > 40:
				#print("\nConfidence: {}\n".format(data['conf'][i]))
				#(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
				#cv2.rectangle(result2, (x, y), (x + w, y + h), (0, 0, 255), 5)
				# Crop the image
				#crp = img_thresh[y:y + h, x:x + w]
				txt = pytesseract.image_to_string(result2, config=config)
					#correct.append('H')
				#print('txts: ',txt)
				result = txt.replace("\n", "")
				txts.append(result)
				conf.append(data['conf'][i])
				#cv2.imshow('rotate', result2)
						#cv2.imshow('TH',img_thresh)
				#cv2.waitKey()
				#cv2.destroyAllWindows()
			else:
				#print("-----------not enough confidence bitch-----------")
				pass
			#cv2.imshow('rotate', result2)
			#cv2.imshow('TH',img_thresh)
			#cv2.waitKey()
			#cv2.destroyAllWindows()


	if len(txts) != 0 and len(conf) != 0:
		#print('cbf lsijiopsa',conf)
		max_conf = max(conf)
		index = conf.index(max_conf)
		alphabet = txts[index]
		#print(alphabet)
	else:
		print("-----------No list here bro!!!!!!!-----------")
		alphabet = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

	return alphabet

def save_img(path,save_path):
	img = cv2.imread(path)
	cv2.imwrite(save_path,img)

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def define_name_colored(name_color):
	name_color = name_color
	if name_color in BLACK:
		name_color = "BLACK"
	elif name_color in WHITE:
		name_color = "WHITE"
	elif name_color in GRAY:
		name_color  = "GRAY"
	elif name_color in RED:
		name_color  = "RED"
	elif name_color in ORANGE:
		name_color  = "ORANGE"
	elif name_color in BROWN:
		name_color  = "BROWN"
	elif name_color in YELLOW:
		name_color  = "YELLOW"
	elif name_color in GREEN:
		name_color  = "GREEN"
	elif name_color in BLUE:
		name_color  = "BLUE"
	elif name_color in PURPLE:
		name_color  = "PURPLE"

	return name_color
#requested_shape_color=find_shape_color('fitcrop/4_5.jpg')
#define_color(requested_shape_color)
#print(convert_rgb_to_names(requested_shape_color))
#print(requested_shape_color[0],requested_shape_color[1],requested_shape_color[2])

#alphabet = text('saved_pic/4_5.jpg')

#find_shape_color('fitcrop/4_5.jpg')
#pos = bg_cut(r'test_pi\img\250.jpg',121)
# shape_color1 = find_shape_color(r"fitcrop\0_0002_square_0.jpg")
# # # #shape_color = define_color(shape_color1)
# apa_color = find_alpha_color(r"fitcrop\0_0002_square_0.jpg")
# # # #alphanumeric_color =  define_color(apa_color)
# actual_name, closest_name = get_colour_name(shape_color1)
# actual_name2, closest_name2 = get_colour_name(apa_color)
# name_color = define_name_colored(closest_name)
# name_color2 = define_name_colored(closest_name2)
# print(closest_name)
# print(closest_name2)
# print(name_color)
# print(name_color2)
#shape_color = define_color(shape_color1)
#alphanumeric_color =  define_color(apa_color)
#bg_cut("D:/work/dataset/img/DSCF5782.JPG",123)
#shapename, name_file, shape_file = shape_name('saved_pic\!_10.jpg')
#hapename, name_file, shape_file = shape_name('saved_pic\!_1.jpg')
#shapename, name_file, shape_file = shape_name('saved_pic\!_11.jpg')