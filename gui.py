from tkinter import *
from tkinter import simpledialog
import tkinter as tk

from PIL import ImageTk, Image, ImageGrab
from itertools import count
import cv2
from PIL import Image
import torch
import threading
# import screeninfo
import base64
import requests
import time
import json                    
import unidecode
import argparse
from gtts import gTTS
import playsound
from datetime import datetime
# import datetime
import os
import numpy as np
import math

from utils.functions import *


from utils.caffe.ultra_face_opencvdnn_inference import inference, net as net_dnn, path
# from mark_detector import MarkDetector
# mark_detector = MarkDetector()

from utils.service.SolvePnPHeadPoseEstimation import * 
from utils.service.TFLiteFaceAlignment import * 
from utils.service.TFLiteFaceDetector import * 
from utils.service.TFLiteIrisLocalization import * 

webcam = cv2.VideoCapture(0)

frame_width = int(webcam.get(3))
frame_height = int(webcam.get(4))


fd = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fa = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")
# hp = HeadPoseEstimator(path + "utils/service/weights/head_pose_object_points.npy", w, h)
hp = HeadPoseEstimator(path + "utils/service/weights/head_pose_object_points.npy", frame_width, frame_height)
gs = IrisLocalizationModel(path + "utils/service/weights/iris_localization.tflite")

# ********************************** Face recognition variables **********************************
parser = argparse.ArgumentParser(description='Face Recognition')
parser.add_argument('-db', '--debug', default='False',
        type=str, metavar='N', help='Turn on debug mode')

args = parser.parse_args()
debug = False
if args.debug == 'True':
	debug = True
global api_list, api_index, url

# url = 'http://10.1.11.47:5051/'
# url = 'http://192.168.0.102:5052/'
url = 'http://192.168.68.120:5052/'

api_list = [url + 'facerec', url + 'FaceRec_DREAM', url + 'FaceRec_3DFaceModeling', url + 'check_pickup']
request_times = [1, 10, 10]
api_index = 0

# test
secret_key = "51bbe3c5-092e-4be4-bcd0-1b438a46b598"

window_name = 'Hệ thống phần mềm AI nhận diện khuôn mặt VKIST'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

temp_boxes = []
temp_ids = []
temp_roles = []
temp_timelines = []
temp_target_index = []

predict_labels = []
protected_boxid = []
queue = []

crop_image_size = 120
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.8
fontcolor = (0,255,0)

extend_pixel = 50
minimum_face_size = 60

box_size = 250
n_box = 1


print('webcam.get(cv2.CAP_PROP_FRAME_WIDTH):', frame_width)
print('webcam.get(cv2.CAP_PROP_FRAME_HEIGHT):', frame_height)

x_dis_image = int((frame_width - box_size * n_box) / (n_box + 1))
y_dis_image = int((frame_height - box_size) / 2)


temp_id = -2
temp_name = 'Unknown'
time_appear = time.time()
max_time_appear = 5
prev_frame_time = 0
new_frame_time = 0

cur_time = 0
max_times = 3

take_photo_state = False
take_sample_data_state = False
take_sample_data_state_GV = False
take_sample_data_state_HS = False
take_sample_data_state_PH = False

verified = False

number_check_points = 5
check_points = (np.zeros(number_check_points) == 1)

center_coordinates = (int(frame_width / 2), int(frame_height / 2))
axesLength = (int(frame_width * 0.2), int(frame_height * 0.35))
angle = 0
startAngle = 0
endAngle = 360

ellipse_points = []
ellipse_points.append([0 + int(frame_width / 2), -axesLength[1] + int(frame_height / 2)])
ellipse_points.append([-axesLength[0] + int(frame_width / 2), 0 + int(frame_height / 2)])
ellipse_points.append([0 + int(frame_width / 2), axesLength[1] + int(frame_height / 2)])
ellipse_points.append([axesLength[0] + int(frame_width / 2), 0 + int(frame_height / 2)])

# sample_face_images = dict()
sample_face_images = []

# print(ellipse_points)

sound_dst_dir = path + 'sounds/'

video_dst_dir = path + 'videos/'




# record_time = datetime.fromtimestamp(time.time())
# year = '20' + record_time.strftime('%y')
# month = record_time.strftime('%m')
# date = record_time.strftime('%d')
# record_time = str(record_time).replace(' ', '_').replace(':', '_')

if not os.path.exists(sound_dst_dir):
    os.makedirs(sound_dst_dir)
# if not os.path.exists(video_dst_dir):
#     os.makedirs(video_dst_dir)
# video_dst_dir += year + '/'
# if not os.path.exists(video_dst_dir):
#     os.makedirs(video_dst_dir)
# video_dst_dir += month + '/'
# if not os.path.exists(video_dst_dir):
#     os.makedirs(video_dst_dir)
# video_dst_dir += date + '/'
# if not os.path.exists(video_dst_dir):
#     os.makedirs(video_dst_dir)

def remove_accent(text):
    return unidecode.unidecode(text)

def set_temp_value(new_id, new_name, is_reset):
	global temp_id, temp_name
	temp_id = new_id
	temp_name = new_name
	if is_reset and debug:
		print("--------------- Reset temp value")
	else:
		if debug:
			print("+++++++++++++++ Update temp value")

def check_first_time_appear(cur_id, cur_name, temp_id_):
	if cur_id != -1:
		if cur_id != temp_id_:
			if debug:
				print('cur_id: ' + str(cur_id) + ' temp_id: ' + str(temp_id_))
			set_temp_value(cur_id, cur_name, False)
			return True
		else:
			return False

def say_hello(content):
    unsign_content = remove_accent(content).replace(" ", "_")
    if not os.path.isfile(sound_dst_dir + unsign_content + ".mp3"):
        if debug:
            print("Creating " + unsign_content + ".mp3 file")
        tts = gTTS(content, tld = 'com.vn', lang='vi')
        tts.save(sound_dst_dir + unsign_content + ".mp3")
    
    playsound.playsound(sound_dst_dir + unsign_content + ".mp3", True)
def face_recognize(frame):
    cur_hour = str(datetime.now()).split(" ")[1].split(":")[0]

    global predict_labels, time_appear, max_time_appear, temp_id, temp_name, cur_time, api_index, max_times
    global temp_ids, temp_roles, temp_timelines

    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps({"secret_key": secret_key, "img": new_img_str})

    seconds = time.time()
    response = requests.post(api_list[api_index], data=payload, headers=headers, timeout=100)

    try:
        print('Server response', response.json())
        for id, time_line, role, bb, name, profileID, generated_face_id in zip(response.json()['result']['id'], response.json()['result']['timelines'], response.json()['result']['roles'], response.json()['result']['bboxes'], response.json()['result']['identities'], response.json()['result']['profilefaceIDs'], response.json()['result']['3DFace'] ):
            response_time_s = time.time() - seconds
            print("Server's response time: " + "%.2f" % (response_time_s) + " (s)")
            bb = bb.split(' ')
            if check_first_time_appear(id, name, temp_id) or api_index == 2:
                time_appear = time.time()
                max_time_appear = 10

                non_accent_name = remove_accent(temp_name)
                if id > -1:
                    front_string = "Xin chào "
                    if int(cur_hour) > 15:
                        front_string = "Tạm biệt "

                    name_parts = temp_name.split(' - ')[0].split(' ')
                    content = ''
                    if non_accent_name.find(' Thi ') > -1 and len(name_parts) < 4:
                        print('\n' + front_string + name_parts[-1] + ' ' + name_parts[0] + '\n')  
                        content = front_string + name_parts[-1] + ' ' + name_parts[0]
                        # say_hello(front_string + name_parts[-1] + ' ' + name_parts[0])
                    else:
                        if len(name_parts) > 2:
                            print(front_string + name_parts[-2] + ' ' + name_parts[-1] + '\n')  
                            # say_hello(front_string + name_parts[-2] + ' ' + name_parts[-1])
                        else:
                            temp_names = ''
                            for name_part_I in name_parts:
                                temp_names += name_part_I + ' '
                            print(front_string + temp_names+ '\n')  
                            # say_hello(front_string + temp_names)

                    faceI = cv2.resize(frame[int(float(bb[1])): int(float(bb[3])), int(float(bb[0])): int(float(bb[2]))], (crop_image_size, crop_image_size))
                    cur_profile_face = None
                    cur_generated_face = None

                    if profileID is not None:
                        cur_url = url + 'images/' + secret_key + '/' + profileID
                        cur_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                        cur_profile_face = cv2.resize(cur_profile_face, (crop_image_size, crop_image_size))
                        cur_profile_face = cv2.cvtColor(cur_profile_face, cv2.COLOR_BGR2RGB)
                    
                    if generated_face_id is not None:
                        cur_url = url + 'images/' + secret_key + '/' + generated_face_id
                        cur_generated_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                        cur_generated_face = cv2.resize(cur_generated_face, (crop_image_size, crop_image_size))
                        cur_generated_face = cv2.cvtColor(cur_generated_face, cv2.COLOR_BGR2RGB)

                    predict_labels.append([non_accent_name, faceI, content, cur_profile_face, cur_generated_face])

                    temp_ids.append(id)
                    temp_timelines.append(time_line)
                    temp_roles.append(role)
            else:
                cur_time += 1
                if cur_time >= max_times:
                    temp_id = -2
                    temp_name = 'Unknown'

    except requests.exceptions.RequestException:
        print(response.text)

    return
# ********************************** Tkinder variables **********************************
x_dis = 10
y_dis = 5

window_size_x = 1200
window_size_y = 700

button_size_x = 120
button_size_x2 = 60
button_size_y = 20

distance_x = 20
distance_y = 30

image_zone_size_x = window_size_x
image_zone_size_y = 640

button_zone_size_x = window_size_x
button_zone_size_y = 50

def mode_1():
    global api_index
    print('\nACTIVATE MODE 1\n')
    take_photo_btn["state"] = DISABLED
    api_index = 0

def mode_2():
    global api_index
    print('\nACTIVATE MODE 2\n')
    take_photo_btn["state"] = DISABLED
    api_index = 1

def take_photo():
    global take_photo_state
    print('\nSend image to 3D face modeling server\n')
    take_photo_state = True

def mode_3():
    global api_index, take_photo_state
    print('\nACTIVATE MODE 3\n')
    take_photo_btn["state"] = NORMAL
    api_index = 2
    take_photo_state = False

def sample_data_GV():
    global take_sample_data_state, check_points, take_sample_data_state_GV, take_sample_data_state_HS, take_sample_data_state_PH
    # print('\nPLEASE ROTATE YOUR HEAD UNTIL ALL DOTS TURN GREEN\n')
    mode_1_btn["state"] = DISABLED
    # mode_2_btn["state"] = DISABLED
    mode_3_btn["state"] = DISABLED
    take_photo_btn["state"] = DISABLED

    take_sample_data_state = True

    take_sample_data_state_GV = True
    take_sample_data_state_HS = False
    take_sample_data_state_PH = False

    sample_data_HS_btn["state"] = DISABLED
    sample_data_PH_btn["state"] = DISABLED

    check_points = (np.zeros(number_check_points) == 1)
    # print('check_points:', check_points) 

def sample_data_HS():
    global take_sample_data_state, check_points, take_sample_data_state_GV, take_sample_data_state_HS, take_sample_data_state_PH
    # print('\nPLEASE ROTATE YOUR HEAD UNTIL ALL DOTS TURN GREEN\n')
    mode_1_btn["state"] = DISABLED
    # mode_2_btn["state"] = DISABLED
    mode_3_btn["state"] = DISABLED
    take_photo_btn["state"] = DISABLED

    take_sample_data_state = True
    
    take_sample_data_state_GV = False
    take_sample_data_state_HS = True
    take_sample_data_state_PH = False

    sample_data_GV_btn["state"] = DISABLED
    sample_data_PH_btn["state"] = DISABLED

    check_points = (np.zeros(number_check_points) == 1)
    # print('check_points:', check_points) 

def sample_data_PH():
    global take_sample_data_state, check_points, take_sample_data_state_GV, take_sample_data_state_HS, take_sample_data_state_PH
    # print('\nPLEASE ROTATE YOUR HEAD UNTIL ALL DOTS TURN GREEN\n')
    mode_1_btn["state"] = DISABLED
    # mode_2_btn["state"] = DISABLED
    mode_3_btn["state"] = DISABLED
    take_photo_btn["state"] = DISABLED

    take_sample_data_state = True
    
    take_sample_data_state_GV = False
    take_sample_data_state_HS = False
    take_sample_data_state_PH = True

    sample_data_GV_btn["state"] = DISABLED
    sample_data_HS_btn["state"] = DISABLED

    check_points = (np.zeros(number_check_points) == 1)
    # print('check_points:', check_points) 

def cancel_data():
    global take_sample_data_state, check_points
    check_points = (np.zeros(number_check_points) == 1)
    take_sample_data_state = False
    mode_1_btn["state"] = NORMAL
    # mode_2_btn["state"] = NORMAL
    mode_3_btn["state"] = NORMAL
    sample_data_GV_btn["state"] = NORMAL
    sample_data_HS_btn["state"] = NORMAL
    sample_data_PH_btn["state"] = NORMAL

def upload_data():
    global take_sample_data_state, check_points, sample_face_images, verified, take_sample_data_state_GV, take_sample_data_state_HS, take_sample_data_state_PH
    is_finish = True
    for pI in check_points:
        if not pI:
            is_finish = False
            break

    if is_finish:
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}
        # user_name = simpledialog.askstring('Username', 'Please insert username')
        # pass_word = simpledialog.askstring('Password', 'Please insert password', show='*')
        # security_payload = json.dumps({"secret_key": secret_key, "username": user_name, "password": pass_word})
        # security_response = requests.post(url + 'verify', data=security_payload, headers=headers, timeout=100)
        # verified = security_response.json()['verified']
        verified = True
        if verified:
            from tkinter import messagebox
 
            # prompt = messagebox.showwarning(title = "Warning!",
            #                                 message = "Warning, errors may occur")

            new_user_id = simpledialog.askstring('New user ID', 'Please insert ID name')
            role = ""
            if take_sample_data_state_GV:
                role = "teacher"
            if take_sample_data_state_HS:
                role = "student"
            if take_sample_data_state_PH:
                role = "parent"

            payload = json.dumps({"secret_key": secret_key, "name": new_user_id, "img": sample_face_images, "type_role":role, "class_id":'-1', "picker_id":'-1', "age":'-1', "gender":'Nam'})

            response = requests.post(url + 'facereg', data=payload, headers=headers, timeout=100)
            print('\nRegister new person is ' + response.json()['result']['message'] + '\n')

            check_points = (np.zeros(number_check_points) == 1)
            take_sample_data_state = False
            take_sample_data_state_GV = False
            take_sample_data_state_HS = False
            take_sample_data_state_PH = False

            mode_1_btn["state"] = NORMAL
            # mode_2_btn["state"] = NORMAL
            mode_3_btn["state"] = NORMAL

            sample_data_GV_btn["state"] = NORMAL
            sample_data_HS_btn["state"] = NORMAL
            sample_data_PH_btn["state"] = NORMAL
            sample_face_images = []
            # print('Complete update data')
            cv2.destroyAllWindows()
        else:
            print("\nWrong username & password\n")

def img_2_str(img):
    _, encimg = cv2.imencode(".jpg", img)
    img_byte = encimg.tobytes()
    return "data:image/jpeg;base64," + base64.b64encode(img_byte).decode('utf-8')

def show_head_pose(euler_angle, face_img):
    """Save face data along with head poses Pitch, Yaw, Roll"""
    global check_points, take_sample_data_state, sample_face_images
    cv2.imshow('test', face_img)
    pitch_angle = euler_angle[0]
    yaw_angle = euler_angle[1]
    roll_angle = euler_angle[2]

    upper_delta_pitch = 5
    upper_delta_yaw = 20
    upper_delta_roll = 10
    lower_delta_pitch = 10
    lower_delta_yaw = 10
    lower_delta_roll = 10

    if pitch_angle < -1 * upper_delta_pitch:
        if not check_points[1]:
            # print('Up')
            # sample_face_images['upFace'] = img_2_str(face_img)
            sample_face_images.append(img_2_str(face_img))
            check_points[1] = True

    if yaw_angle > upper_delta_yaw:
        if not check_points[2]:
            # print('Right')
            # sample_face_images['rightFace'] = img_2_str(face_img)
            sample_face_images.append(img_2_str(face_img))
            check_points[2] = True
    
    if pitch_angle > upper_delta_pitch:
        if not check_points[3]:
            # print('Down')
            # sample_face_images['downFace'] = img_2_str(face_img)
            sample_face_images.append(img_2_str(face_img))
            check_points[3] = True
    
    if yaw_angle < -1 * upper_delta_yaw:
        if not check_points[4]:
            # print('Left')
            # sample_face_images['leftFace'] = img_2_str(face_img)
            sample_face_images.append(img_2_str(face_img))
            check_points[4] = True

    if abs(pitch_angle) < lower_delta_pitch and abs(yaw_angle) < lower_delta_yaw and abs(roll_angle) < lower_delta_roll:
        if not check_points[0]:
            # print('Frontal')
            # sample_face_images['frontalFace'] = img_2_str(face_img)
            sample_face_images.append(img_2_str(face_img))
            check_points[0] = True

def mode_4():
    global api_index
    print('\nACTIVATE MODE 4\n')
    take_photo_btn["state"] = DISABLED

# def get_y_in_Ellipse(x_center, y_center, axis_x, axis_y, x, angle = 0):
    
#     # # WITH StackOverflow Answer Edits
#     angle_rad = math.radians(angle)
#     cosa = math.cos(angle_rad)
#     sina = math.sin(angle_rad)

#     # # Equation of point within angled ellipse
#     a = (((cosa * (x_center - x) + sina * (y_center - y)) ** 2) / (axis_x**2))
#     b = (((sina * (x_center - x) - cosa * (y_center - y)) ** 2) / (axis_y**2))

#     return []

root = Tk()
root.title(window_name)
root.geometry(str(window_size_x) + 'x' + str(window_size_y))

window_position_x = int((root.winfo_screenwidth() - window_size_x) / 2)
root.geometry('+{}+{}'.format(window_position_x,0))

# Create image name zone
image_name_zone = Canvas(root, width = image_zone_size_x, height = 40, bg="white")
image_name_zone.grid(row=0, column=0)

# Create image zone
image_zone = Canvas(root, width = image_zone_size_x, height = image_zone_size_y - 40, bg="white")
image_zone.grid(row=1, column=0)

image_id = None  # default value at start (to create global variable)

# Create button zone
button_zone = Frame(root, width = button_zone_size_x, height = button_zone_size_y, bg='#bbbcbd')
button_zone.place(x= 0, y = image_zone_size_y + y_dis)

var = IntVar()
# Create a Button
mode_1_btn = Radiobutton(button_zone, text = 'Chế độ 1', variable=var, value=1, bd = '5', command = mode_1)
mode_1_btn.place(x = x_dis, y = y_dis * 2)
mode_1_btn.select()

# mode_2_btn = Radiobutton(button_zone, text = 'Chế độ 2', variable=var, value=2, bd = '5', command = mode_2)
# mode_2_btn.place(x = button_size_x + x_dis, y = y_dis * 2)

mode_3_btn = Radiobutton(button_zone, text = 'Chế độ 3', variable=var, value=3,  bd = '5', command = mode_3)
mode_3_btn.place(x = button_size_x * 1 + x_dis, y = y_dis * 2)

take_photo_btn = Button(button_zone, text = 'Chụp ảnh!', bd = '5', command = take_photo)
take_photo_btn.place(x = button_size_x * 2 + x_dis, y = y_dis * 2)

take_photo_btn["state"] = DISABLED

sample_data_GV_btn = Button(button_zone, text = 'ĐK GV', bd = '5', command = sample_data_GV)
sample_data_GV_btn.place(x = button_size_x * 3 + x_dis, y = y_dis * 2)

sample_data_HS_btn = Button(button_zone, text = 'ĐK HS', bd = '5', command = sample_data_HS)
sample_data_HS_btn.place(x = button_size_x * 3 + button_size_x2 + x_dis, y = y_dis * 2)

sample_data_PH_btn = Button(button_zone, text = 'ĐK PH', bd = '5', command = sample_data_PH)
sample_data_PH_btn.place(x = button_size_x * 3 + button_size_x2 * 2 + x_dis, y = y_dis * 2)

cancel_data_btn = Button(button_zone, text = 'Hủy mẫu', bd = '5', command = cancel_data)
cancel_data_btn.place(x = button_size_x * 3 + button_size_x2 * 3 + x_dis * 2, y = y_dis * 2)

upload_data_btn = Button(button_zone, text = 'Đăng ký', bd = '5', command = upload_data)
upload_data_btn.place(x = button_size_x * 4 + button_size_x2 * 3 + x_dis, y = y_dis * 2)

# sample_data_btn["state"] = DISABLED
count = 0

def largest_indices(arr):
    first_largest = second_largest = float("-inf")
    first_index = second_index = None

    for i, num in enumerate(arr):
        if num > first_largest:
            second_largest = first_largest
            second_index = first_index

            first_largest = num
            first_index = i
        elif num > second_largest:
            second_largest = num
            second_index = i

    return first_index, second_index

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = frame_width
        self.height = frame_height
        self.interval = 20 # Interval in ms to get the latest frame
        self.prev_frame_time = 0
        self.new_frame_time = 0

        # Create canvas for image
        # self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        # self.canvas.grid(row=0, column=0)

        image_names = ['Ảnh tức thời', 'Ảnh bé', 'Ảnh người đón']
        for i, name_I in enumerate(image_names):
            image_name_zone.create_text(frame_width + distance_x + (crop_image_size + distance_x) * i + 60, int(distance_y * 0.5), text=name_I, fill="black", font=('Helvetica 15 bold'))
        
        # Update image on canvas
        self.update_image()

    def update_image(self):
        global count, predict_labels, temp_boxes, prev_frame_time, new_frame_time, queue, api_index, request_times, take_photo_state, protected_boxid, take_sample_data_state, temp_target_index
        global image_id, temp_ids, temp_roles, temp_timelines, extend_pixel
        count += 1

        frame_show = np.ones((window_size_y, window_size_x, 3),dtype='uint8') * 255    

        ret, orig_image = self.cap.read()
        orig_image = cv2.flip(orig_image, 1)

        final_frame = orig_image.copy()

        if take_sample_data_state:
            face_region_mask = np.zeros((self.height, self.width), np.uint8)
            remain_region_mask = np.ones((self.height, self.width), np.uint8)

            cv2.ellipse(face_region_mask, center_coordinates, axesLength, angle, startAngle, endAngle, 1, thickness = -1)
            cv2.ellipse(remain_region_mask, center_coordinates, axesLength, angle, startAngle, endAngle, 0, thickness = -1)

            # face_region = cv2.bitwise_and(orig_image, orig_image, mask = face_region_mask)
            # blur_remain_region = cv2.bitwise_and(cv2.blur(orig_image,(100, 100)), cv2.blur(orig_image,(100, 100)), mask = remain_region_mask)

            # final_frame = blur_remain_region + face_region
            cv2.putText(final_frame, ('Only ONE person each time!'), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)

            # Draw points
            for i, ellipse_point_I in enumerate(ellipse_points):
                if check_points[i + 1]:
                    cv2.circle(final_frame, (ellipse_point_I[0], ellipse_point_I[1]), 10, (0, 255, 0), -1)  
                else:
                    cv2.circle(final_frame, (ellipse_point_I[0], ellipse_point_I[1]), 10, (0, 0, 255), -1)  

            temp_boxes, _ = fd.inference(orig_image)
            if len(temp_boxes) == 0:
                temp_boxes, _, probs = inference(net_dnn, orig_image)

            delta = 40
            if len(temp_boxes) == 1:
                xmin, ymin, xmax, ymax = int(temp_boxes[0][0]), int(temp_boxes[0][1]), int(temp_boxes[0][2]), int(temp_boxes[0][3])
                xmin -= extend_pixel
                xmax += extend_pixel
                ymin -= 2 * extend_pixel
                ymax += extend_pixel

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = frame_width if xmax >= frame_width else xmax
                ymax = frame_height if ymax >= frame_height else ymax

                marks = fa.get_landmarks(orig_image, temp_boxes)
                for landmarks, bbox_I in zip(marks, temp_boxes):
                    # calculate head pose Pitch, Yaw, Roll
                    euler_angle = hp.get_head_pose(landmarks).flatten()
                    # print(euler_angle)
                    pose_name = ['Pitch', 'Raw', 'Roll']            
                    for j in range(len(euler_angle)):
                        cv2.putText(final_frame, (pose_name[j] + ': {:05.2f}').format(float(-1 * euler_angle[j])), (10, frame_height - 20 - (40 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
                    show_head_pose(euler_angle, orig_image[ymin:ymax, xmin:xmax])

        else:
            temp_boxes, _ = fd.inference(orig_image)

            if len(temp_boxes) == 0:
                temp_boxes, _, probs = inference(net_dnn, orig_image)

            box_dimensions = []
            for i, bbox_I in enumerate(temp_boxes):

                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
                diagonal = math.sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin))
                box_dimensions.append(diagonal)

            if len(temp_boxes) > 1:
                temp_target_index = largest_indices(box_dimensions)

            # draw_box(final_frame, temp_boxes, box_color=(0, 255, 0))
            marks = fa.get_landmarks(orig_image, temp_boxes)

            if (api_index < 2 or (api_index == 2 and take_photo_state)):
                if (count % request_times[api_index]) == 0:
                    
                    for landmarks, bbox_I in zip(marks, temp_boxes):

                        xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                        # draw_box(final_frame, [[xmin, ymin, xmax, ymax]], box_color=(0, 0, 255))
                        # if api_index == 2 and take_photo_state:
                        xmin -= extend_pixel
                        xmax += extend_pixel
                        ymin -= 2 * extend_pixel
                        ymax += extend_pixel
                        draw_box(final_frame, [[xmin, ymin, xmax, ymax]], box_color=(255, 0, 0))

                        xmin = 0 if xmin < 0 else xmin
                        ymin = 0 if ymin < 0 else ymin
                        xmax = frame_width if xmax >= frame_width else xmax
                        ymax = frame_height if ymax >= frame_height else ymax

                        # for index, idI in enumerate(landmarks):
                        #     cv2.circle(final_frame, (int(marks[index][0]), int(marks[index][1])), 5, (0, 0, 255), -1)  

                        queue = [t for t in queue if t.is_alive()]
                        if len(queue) < 3:
                            queue.append(threading.Thread(target=face_recognize, args=(orig_image[ymin:ymax, xmin:xmax],)))
                            # queue.append(threading.Thread(target=face_recognize, args=(orig_image)))
                            queue[-1].start()
                        count = 0
                    take_photo_state = False
                    
                    # Check pickup process
                    if len(temp_boxes) > 1:
                        if len(temp_ids) < 2 or len(temp_ids) < 2 or len(temp_ids) < 2:
                            print('\nPick up fail\n')
                        else: 
                            # print('temp_target_index:', temp_target_index)
                            index1 = temp_target_index[0]
                            index2 = temp_target_index[1]
                            check_pickup_info = {}

                            # print('temp_ids:', temp_ids)
                            # print('temp_timelines:', temp_timelines)
                            # print('temp_roles:', temp_roles)
                            if (temp_roles[index1] == 'student' and temp_roles[index2] != 'student'):
                                check_pickup_info = {
                                    'student_id': temp_ids[index1],
                                    'picker_id': temp_ids[index2],
                                    'timeline_student': temp_timelines[index1],
                                    'timeline_picker': temp_timelines[index2],
                                    'secret_key': secret_key
                                }
                            if (temp_roles[index2] == 'student' and temp_roles[index1] != 'student'):
                                check_pickup_info = {
                                    'student_id': temp_ids[index2],
                                    'picker_id': temp_ids[index1],
                                    'timeline_student': temp_timelines[index2],
                                    'timeline_picker': temp_timelines[index1],
                                    'secret_key': secret_key
                                }
                            if len(check_pickup_info) > 0:
                                # print('check_pickup_info:', check_pickup_info)
                                headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}
                                payload = json.dumps(check_pickup_info)
                                response = requests.post(api_list[3], data=payload, headers=headers, timeout=100)
                                print('\nPick up ' + response.json()['result']['message'] + '\n')
                            temp_ids = []
                            temp_timelines = []
                            temp_roles = []
        image_name_y = 5
        temp_labels = list(reversed(predict_labels))
        for i, labelI in enumerate(temp_labels):
            if frame_width + distance_x + crop_image_size < window_size_x and int((crop_image_size + distance_y) * i) + distance_y + crop_image_size < window_size_y:
                cv2.putText(frame_show, '{0}'.format(labelI[0]), (frame_width + distance_x, int((crop_image_size + distance_y) * i) + int(distance_y / 1.5)  + image_name_y), fontface, fontscale, (100, 255, 0))

                frame_show[int((crop_image_size + distance_y) * i) + distance_y + image_name_y: int((crop_image_size + distance_y) * i) + distance_y + image_name_y + crop_image_size, frame_width + distance_x: frame_width + distance_x + crop_image_size, :] = labelI[1]

                if labelI[3] is not None:
                    frame_show[int((crop_image_size + distance_y) * i) + distance_y + image_name_y: int((crop_image_size + distance_y) * i) + distance_y + image_name_y + crop_image_size, frame_width + distance_x * 2 + crop_image_size: frame_width + distance_x * 2 + crop_image_size * 2, :] = labelI[3]
                
                if labelI[4] is not None:
                    frame_show[int((crop_image_size + distance_y) * i) + distance_y + image_name_y: int((crop_image_size + distance_y) * i) + distance_y + image_name_y + crop_image_size, frame_width + distance_x * 3 + crop_image_size * 2: frame_width + distance_x * 3 + crop_image_size * 3, :] = labelI[4]
        
        frame_show[:frame_height, :frame_width,:] = final_frame

        if len(predict_labels) > 4:
            predict_labels = predict_labels[1:]

        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        fps = str(int(fps))

        cv2.putText(frame_show, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        self.image = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB) # to RGB

        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format

        # Update image
        image_zone.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)

if __name__ == "__main__":
    MainWindow(root, webcam)
    root.mainloop()
