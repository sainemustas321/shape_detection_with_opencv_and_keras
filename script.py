import os
import shutil
import ppp as ppp
import glob
import writeJSON as writeJSON
import json
import time
shapename = []
name_file= []
shape_file = []
pos = []
check_positions = []

#file_check = glob.glob('Z:/test/img/*.jpg')
shapepath = glob.glob('D:/work/avengers_assemble/saved_pic/*.jpg')
path_color = glob.glob('D:/work/avengers_assemble/fitcrop/*.jpg')

def copyy(source, destination):
	err_host = False
	try:
		shutil.copyfile(source, destination)
		print("File copied successfully.")
 
	# If source and destination are same
	except shutil.SameFileError:
		print("Source and destination represents the same file.")
	
	# If destination is a directory.
	except IsADirectoryError:
		print("Destination is a directory.")
	
	# If there is any permission issue
	except PermissionError:
		print("Permission denied.")
	
	# For other errors
	except Exception as e:
		print("Error occurred while copying file.")
		print(e, type(e))
		
		if 'Host is down' in str(e):
			print('Host is down, waiting it')
			err_host = True
	while err_host:
		print('try copy...')
		try:
			shutil.copyfile(source, destination)
			break
		except Exception as e:
			if 'Host is down' in str(e):
				continue
			else:
				break
			

APP_FOLDER = 'Z:/test_pi/data' #'Z:/test_pi/data'
APP_FOLDER2 = 'Z:/test_pi/img'#'Z:/test_pi/img'
APP_FOLDER3 = 'D:/work/avengers_assemble/saved_pic'
#APP_FOLDER4 = 'D:/work/avengers_assemble/fitcrop'
APP_FOLDER5 = 'D:/work/avengers_assemble/shape'

APP_FOLDER6 = 'D:/work/avengers_assemble/JSON_AUTO'
APP_FOLDER7 = 'D:/work/avengers_assemble/JSON_MANUAL'




totalFiles = 0
totalDir = 0

count_crop = 0
checkname = []
file_in = 0
list_file = []
list_file2 = []
list_file5 = []
list_file3 = []


input("Enter to start")

# list_file = os.listdir(APP_FOLDER2)
# list_file2 = os.listdir(APP_FOLDER)
# list_file5 = os.listdir(APP_FOLDER5)
# list_file3 = os.listdir(APP_FOLDER3)

print(os.listdir(APP_FOLDER))
stt = time.time()

tr_number = int(input("Target__: "))
while True:
	
	try:

		if len(os.listdir(APP_FOLDER)) != file_in:
			
			print("start get")
			#print(os.listdir(APP_FOLDER))
			file_in = len(os.listdir(APP_FOLDER))
			
			for _name , _name2  in zip(os.listdir(APP_FOLDER2), os.listdir(APP_FOLDER)):
				
				if not '.jpg' in _name.lower():
					continue

				if not _name in list_file and not _name2 in list_file2:
					#print("----------------Check ERROR----------------")
					ddd = ppp.source(APP_FOLDER+'/'+_name2)
					heading = ddd[4]

					if heading == "404":
						print("---------------- ควยไอ้เหี้ย ERROR 404!!!!----------------")
						pass

					else:
						list_file.append(_name)
						list_file2.append(_name2)
						if tr_number >= int(_name.split('.jpg')[0]):
							continue
						print(f"Do it file {_name}...")
						print(f"Do it txt {_name2}...")
						print("----------------ตัดหลังละไอ้เหี้ย----------------")
						pos = ppp.bg_cut(APP_FOLDER2 +f'/{_name}', count_crop)
						count_crop += 1

						ff_in = 0
						
						
					# while True:
							# if len(os.listdir(APP_FOLDER3)) != ff_in:
							#     print("start get")
							#     print(os.listdir(APP_FOLDER3))
						#
						_n5 = 0
						for _name3 in os.listdir(APP_FOLDER3):
								if not _name3 in list_file3:# and not _name4 in list_file4: 
									print("----------------ได้สักทีไอ้เหี้ย(หาร่างละไอ้หน้าหี)----------------")
									list_file3.append(_name3)  
									  
									#list_file4.append(_name4) 
									
									print(f"Do it picture {_name3}...")
									#print(f"Do it mask-roi {_name4}...")
									
									#print(shapename, name_file, shape_file)

									if len(pos) != 0:                              
										shapename, name_file, shape_file = ppp.shape_name(f'D:/work/avengers_assemble/mask_roi/{_name3}', f'D:/work/avengers_assemble/saved_pic/{_name3}') #3
										
										#print(shapename)
										shapename = shapename
										
										for _name5 in os.listdir(APP_FOLDER5):
											#print(name_file, checkname ,_name5, APP_FOLDER5)
											
											if not _name5 in list_file5 : 
												list_file5.append(_name5)
												print("----------------ได้เถอะสัด(เข้าสู่การจวรตตตต)----------------") 

												shape1_color = ppp.find_shape_color(f'D:/work/avengers_assemble/fitcrop/{_name5}') #4
												alpha_color = ppp.find_alpha_color(f'D:/work/avengers_assemble/fitcrop/{_name5}') #3
												aaa ,shape_color111 = ppp.get_colour_name(shape1_color)
												bbbb ,alphanumeric_color111 = ppp.get_colour_name(alpha_color)
												print(shape_color111, alphanumeric_color111)
												shape_color = ppp.define_name_colored(shape_color111)
												alphanumeric_color = ppp.define_name_colored(alphanumeric_color111)
												alphabet = ppp.text(f'D:/work/avengers_assemble/shape/{_name5}') #3

												data = ppp.source(APP_FOLDER+'/'+_name2)
												print(_name5, pos[_n5])
												#print("opkopdkqwopdkoqwpdqw", pos)
												new_lat, new_long, heading = ppp.lat_long(float(data[1]),float(data[2]),float(data[3]) ,float(data[4]), pos[_n5])
												print(new_lat, new_long)
												writeJSON.write(f'D:/work/avengers_assemble/shape/{_name5}',str(_name5),shapename,shape_color,alphanumeric_color,alphabet,heading,new_lat,new_long)#3
												_n5+=1
												#checkname.append(f'D:/work/avengers_assemble/shape/{_name5}')#3
												#ppp.save_img(f'D:/work/avengers_assemble/shape/{_name5}.jpg','D:/work/avengers_assemble/ans_auto/')
											# json_auto = []
											# json_manual = []

											# for _name6, _name7 in zip(os.listdir(APP_FOLDER6), os.listdir(APP_FOLDER7)):

											# 	if not json_auto in APP_FOLDER6 and not json_manual in APP_FOLDER7:
											# 		json_auto.append(_name6)
											# 		json_manual.append(_name7)
											# 		print(f"read json auto from {_name6}")
											# 		print(f"read json manual from {_name7}")
											# 		print("------------CHECK JSON AUTO WITH MANNUAL------------")
											# 		a, b = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
											# 		print('JSON AUTO: ',a)
											# 		print('JSON MANUAL: ',b)
											# 		if a == b :
											# 			print("------------คำตอบถูกนะไอ้สัด------------")
											# 		else:
											# 			pass
												#input("rECH")

			else:
				print("----------------ควยไอ้เหี้ย ERROR 404 NOT FOUND----------------")
				pass
			# for i in range(len(os.listdir(APP_FOLDER))-file_in):
			#     print("do it!")
			
			print("total : ", time.time()-stt)
			print("Next...")

	except Exception as e: # generaly not a great idea, put a more refined exception
			print(e)
			continue




