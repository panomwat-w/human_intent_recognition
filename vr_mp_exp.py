import os
import numpy as np
import pybullet as p
import pybullet_data
import threading
import json
import socket
# import tensorflow as tf

from task_environment import *
import time

# # tf.config.set_visible_devices([], 'GPU')
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

localIP = "127.0.0.1"
localPort = 12312
msgFromServer = ["Hello UDP Client"]
lock = threading.Lock()

pos_orient = [0]*7
gaze_pos = [0,0,0]
pointRay = -1
# project_path = "C:\Users\birl\dissertation\human_intent_recognition"
project_path = os.getcwd()

CONTROLLER_ID = 0
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")

def getting_phantom_pos(UDPServerSocket, msgFromServer, terminate, bufferSize = 1024):
    global pos_orient
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        dict_msg = json.loads(message)

        pos_orient = dict_msg["pos_orin"]

        bytesToSend = str.encode(msgFromServer[0])
        UDPServerSocket.sendto(bytesToSend, address)

        if terminate.is_set():
            break

def initialise():

    ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'),)
    camera = Camera((0.12, -0.1, 1.5),
                    (0.12, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (160, 160), 40)
    camera = None
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))

    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
        p.connect(p.GUI)
        # p.setInternalSimFlags(0) 
    
    p.resetSimulation()
    p.setAdditionalSearchPath(project_path)
    
    env = ClutteredPushGraspVR(robot, ycb_models, camera, vis=True)  
    env.reset()
    env.SIMULATION_STEP_DELAY = 1e-4
    vr_camera_pos = [0,-1.9,0.09]
    vr_camera_orn_euler = [-np.pi/2 + np.pi/15,0,0]
    vr_camera_orn_euler = [-np.pi/4,0,0]
    vr_camera_orn = p.getQuaternionFromEuler(vr_camera_orn_euler)
    p.setVRCameraState(vr_camera_pos, vr_camera_orn)
    for i in range(3):
        warm_up(env)
    brick_id, remove_brick_origin = initialize_brick(env)

    return env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler

def initialize_brick(env):
    
    fix_brick_origin, remove_brick_origin = calc_brick_origin(7, (0.0, -0.1), (0.04, 0.08), 0.015, 0.02)
    fix_brick_id_list = []
    for i in range(len(fix_brick_origin)):
        brick_id = p.loadURDF("meshes/brick/brick_clone.urdf", fix_brick_origin[i], useFixedBase=True) 
        env.object_ids.append(brick_id)
        fix_brick_id_list.append(brick_id)
    brick_origin = (-0.18, -0.1, 0.02) 
    brick_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
    # brick_orn = p.getQuaternionFromEuler([0, 0, 0])
    brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, brick_orn, useFixedBase=False) 
    p.changeDynamics(brick_id, -1, restitution=0.2, frictionAnchor=1, lateralFriction=2)
    env.object_ids.append(brick_id)
    wall_origin = (-0.05, -0.05, 0.1) 
    wall_id = p.loadURDF("meshes/brick/wall.urdf", wall_origin, useFixedBase=True) 
    env.object_ids.append(wall_id)
    
    return brick_id, remove_brick_origin

def get_gaze_pos(terminate):
    global gaze_pos
    global pointRay
    while True:
        events = p.getVREvents(p.VR_DEVICE_HMD)
        
        for e in (events):
            pos = e[POSITION]
            orn = e[ORIENTATION]
            lineFrom = pos
            mat = p.getMatrixFromQuaternion(orn)
        #   dir = [mat[0], mat[3], mat[6]]
        #   dir = [mat[1], mat[4], mat[7]]
            dir = [-mat[2], -mat[5], -mat[8]]
        #   hit = p.rayTest(lineFrom, to)

            length = - pos[2] / dir[2]
            # oldRay = pointRay
            if length > 0:
                gaze_pos = [pos[0] + dir[0] * length, pos[1] + dir[1] * length, 0.0]
                lineFrom = [gaze_pos[0] + 0.005, gaze_pos[1] + 0.005, 0.005]
                color = [0, 0, 1]
                width = 5
                pointRay = p.addUserDebugLine(lineFrom,gaze_pos,color,width, replaceItemUniqueId=pointRay)
        if terminate.is_set():
            break

def get_vr_button(env):
    while True:
        events = p.getVREvents(allAnalogAxes=1)
        for e in (events):
            if e[CONTROLLER_ID] == 2:
                if (e[BUTTONS][1] & p.VR_BUTTON_WAS_TRIGGERED):
                    return 0
                elif (e[BUTTONS][33] & p.VR_BUTTON_WAS_TRIGGERED):
                    return 1
                elif (e[BUTTONS][7] & p.VR_BUTTON_WAS_TRIGGERED):
                    return 2
                elif (e[BUTTONS][2] & p.VR_BUTTON_WAS_TRIGGERED):
                    return 3
                elif (e[BUTTONS][32] & p.VR_BUTTON_WAS_TRIGGERED): 
                    return 4
            else:
                if (e[BUTTONS][32] & p.VR_BUTTON_WAS_TRIGGERED): 
                    return -1
                elif (e[BUTTONS][1] & p.VR_BUTTON_WAS_TRIGGERED):
                    warm_up(env)


def inference_udp(pred_threshold=0.6):
    global msgFromServer
    
    max_len = 170
    motion_list = []
    prev_pos_orient = pos_orient
    state = 0
    # debug_text = -1
    debug_text = p.addUserDebugText("Ready", (0.2,0.1,0.3), (1,0,0))
    while True:
        # print(pos_orient)
        events = p.getVREvents()
        for e in events:
            if (e[BUTTONS][7] & p.VR_BUTTON_WAS_TRIGGERED):
                return -1

        if state == 0 and pos_orient[6] == 1:
            state = 1
            print("Button pressed, start recording ...")
            with lock:
                msgFromServer[0] = "Push"
            p.removeUserDebugItem(debug_text)
            debug_text = p.addUserDebugText("Button pressed, start recording ...", (0.2,0.1,0.3), (1,0,0))

        if state == 1 and pos_orient[6] == 1:
            with lock:
                msgFromServer[0] = "Push"
            motion_list.append(pos_orient)
            time.sleep(0.01)

        if state == 1 and pos_orient[6] == 0:
            state = 0
            with lock:
                msgFromServer[0] = "Hello UDP Client"
            p.removeUserDebugItem(debug_text)
            debug_text = p.addUserDebugText("Button released", (0.2,0.1,0.3), (1,0,0))
            print("Button released")
            print("length of motion data : ", len(motion_list))
            if len(motion_list) == 0:
                print("No motion data, please try again")
                motion_list = []
                print("Ready for new data")
                continue
            if len(motion_list) > max_len:
                print("Too long, motion data is truncated")
                motion_list = motion_list[:max_len]
            
            pred_dict = receive_message(motion_list)
            pred = np.array(pred_dict['pred'])
            pred_class = pred.argmax()
            print(f"Predicted Class = {pred_class}")
            p.removeUserDebugItem(debug_text)
            if pred.max() < pred_threshold:
                print("Low confidence, please try again")
                motion_list = []
                print("Ready for new data")
                continue
            else:
                print("High confidence, executing motion primitives")
                return pred_class

def get_pos_orient():
    return pos_orient            

def user_control_demo(env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler):
    fname = input("Enter log file name <name>_<date>: ")
    fname += "_mp_2"
    terminate = threading.Event()
    x = threading.Thread(target=getting_phantom_pos, args=(UDPServerSocket, msgFromServer,terminate,))
    x.start()
    y = threading.Thread(target=get_gaze_pos, args=(terminate,))
    y.start()
    adjust_camera(vr_camera_pos, vr_camera_orn_euler)
    # check_connection(env, pos_orient)
    break_loop = False
    for i in range(1000):
        print(pos_orient)
        events = p.getVREvents()
        for e in events:
            if (e[BUTTONS][7] & p.VR_BUTTON_WAS_TRIGGERED):
                break_loop = True
        if break_loop:
            break                
    # model_name="model/new/best_lstm_model.h5"
    # model = load_model(model_name)
    # for i in range(3):
    #     warm_up(env)
    # current_coordinate = env.robot.get_coordinate()
    # current_pos = np.array(current_coordinate[0])
    # current_orn = np.array(p.getEulerFromQuaternion(current_coordinate[1]))
    # print(current_pos, current_orn)
    
    # pointRay = -1
    state = 0
    count = 0
    grip_attempt = 0
    start = time.time()
    z = threading.Thread(target=log_robot_object, args=(env, fname, brick_id, start, terminate, get_pos_orient, remove_brick_origin,))
    z.start()
    cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
    while True:
        # remove_all_objects(env)
        # gaze_pos, pointRay = get_gaze_pos(pointRay)
        
        # pred_class = get_vr_button(env)
        pred_class = inference_udp()
        if pred_class == 0:
            grip_attempt += 1
        # p.removeUserDebugItem(debug_text)
        
        if pred_class == -1:
            break
        # pred_class = count % 3
        # count += 1
        obj_pos = [gaze_pos[0], gaze_pos[1], gaze_pos[2]+0.21]
        d_obj = calculate_distance(np.array(obj_pos[:2]), np.array(cubePos[:2]))
        if d_obj < 0.05 :
            obj_pos[:2] = cubePos[:2]
            obj_pos[2] = cubePos[2] + 0.21
        target_pos=[gaze_pos[0], gaze_pos[1], 0.28]
        d_target = calculate_distance(np.array(target_pos[:2]), np.array(remove_brick_origin[:2]))
        if d_target < 0.05 :
            target_pos = remove_brick_origin
        obs, _, _, _ = apply_motion_primitives(env, pred_class, obj_pos=obj_pos, des_pos=target_pos)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)

        d = calculate_distance(np.array(cubePos[:2]), np.array(remove_brick_origin[:2]))
        if d < 0.05 and cubePos[2] < 0.05:
            # debug_text = p.addUserDebugText("Successful", (0.2,0.1,0.3), (0,1,0))
            # time.sleep(0.5)
            # p.removeUserDebugItem(debug_text)
            print(f'Grasping Attempts: {grip_attempt}')
            break
    terminate.set()
    # y.join()
    # x.join()
          

if __name__ == '__main__':
    env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler = initialise()
    user_control_demo(env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler)
    # burn_samples()
    env.close()
