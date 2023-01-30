import numpy as np
import math
import cv2
import struct as st
from numpy import dot, sqrt
from numpy.linalg import norm

def draw_box(image, boxes, box_color=(255, 255, 255), thickness = 3):
    """Draw square boxes on image"""
    for box in boxes:
        cv2.rectangle(image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), box_color, 3)

def draw_circle(image, boxes, box_color=(255, 255, 255), thickness = 3):
    """Draw square boxes on image"""
    for box in boxes:
        radius = int(sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2) * 0.4)
        cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), radius, box_color, 3)

def draw_ellipse(image, boxes, box_color=(255, 255, 255), thickness = 3):
    """Draw square boxes on image"""
    for box in boxes:
        center_coordinates = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        axesLength = (int((box[2] - box[0]) * 0.5), int((box[3] - box[1]) * 0.5))
        angle = 0
        startAngle = 0
        endAngle = 360
        cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, box_color, thickness)


def cosine_similarity(a, b):
	return dot(a, b)/(norm(a)*norm(b))

def euclidean_distance(a, b):
    dist = np.linalg.norm(a-b)
    return dist

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(5 * (abs(angle) / 45.0 - 1))
    return norm_angle

def face_orientation(frame, landmarks):
    # print(landmarks)
    size = frame.shape #(height, width, color_channel)
    # index_6_point = [36, 45, 30, 8, 48, 54]
    image_points = np.array([
                            (landmarks[30][0], landmarks[30][1]),     # Nose tip
                            (landmarks[8][0], landmarks[8][1]),       # Chin
                            (landmarks[36][0], landmarks[36][1]),     # Left eye left corner
                            (landmarks[45][0], landmarks[45][1]),     # Right eye right corne
                            (landmarks[48][0], landmarks[48][1]),     # Left Mouth corner
                            (landmarks[54][0], landmarks[54][1])      # Right mouth corner
                        ], dtype="double")
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)#, flags=cv2.CV_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (image_points[0][0], image_points[0][1]), image_points
def get_poliface_angle(cam_name):
    cam_name = cam_name.split('_')[0]
    roll = 0
    pitch = 0
    yaw = 360

    if cam_name == 'C1':
        yaw = -90
    elif cam_name == 'C2':
        yaw = -75
    elif cam_name == 'C3':
        yaw = -60
    elif cam_name == 'C4' or cam_name == 'C14' or cam_name == 'C21':
        yaw = -45
    elif cam_name == 'C5' or cam_name == 'C15' or cam_name == 'C22':
        yaw = -30
    elif cam_name == 'C6' or cam_name == 'C16' or cam_name == 'C23':
        yaw = -15
    elif cam_name == 'C7' or cam_name == 'C17' or cam_name == 'C24':
        yaw = 0
    elif cam_name == 'C8' or cam_name == 'C18' or cam_name == 'C25':
        yaw = 15
    elif cam_name == 'C9' or cam_name == 'C19' or cam_name == 'C26':
        yaw = 30
    elif cam_name == 'C10' or cam_name == 'C20' or cam_name == 'C27':
        yaw = 45
    elif cam_name == 'C11':
        yaw = 60
    elif cam_name == 'C12':
        yaw = 75
    elif cam_name == 'C13':
        yaw = 90
    
    if cam_name == 'C14'or cam_name == 'C15' or cam_name == 'C16' or cam_name == 'C17'or cam_name == 'C18'or cam_name == 'C19'or cam_name == 'C20':
        pitch = 30
    if cam_name == 'C21'or cam_name == 'C22' or cam_name == 'C23' or cam_name == 'C24'or cam_name == 'C25'or cam_name == 'C26'or cam_name == 'C27':
        pitch = -15

    return roll, pitch, yaw

def load_feat(feat_file):
    feats = list()
    with open(feat_file, 'rb') as in_f:
        feat_num, feat_dim = st.unpack('ii', in_f.read(8))
        for i in range(feat_num):
            feat = np.array(st.unpack('f'*feat_dim, in_f.read(4*feat_dim)))
            feats.append(feat)
    feats = np.array(feats)
    return feats

class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count