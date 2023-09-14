import os
import numpy as np
import pybullet as p
import pybullet_data
import threading
import json
import socket
import tensorflow as tf

from task_environment import *
import time

# # tf.config.set_visible_devices([], 'GPU')
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
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
    # vr_camera_pos = [1.5,0.5,0.09]
    # vr_camera_pos = [1.46, -0.12000000000000012, 0.09] 
    vr_camera_orn_euler = [-0.7853981633974483, -0.08168140899333459, 1.4011503235010618]
    # vr_camera_orn_euler = [-np.pi/2 + np.pi/15,0,0]
    vr_camera_orn_euler = [-np.pi/4,0,0]
    vr_camera_orn = p.getQuaternionFromEuler(vr_camera_orn_euler)
    p.setVRCameraState(vr_camera_pos, vr_camera_orn)
    for i in range(1):
        warm_up(env)
    brick_id, remove_brick_origin = initialize_brick(env)

    return env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler

def initialize_brick(env):
    brick_origin = (-0.3, -0.05, 0.02) 
    fix_brick_origin, remove_brick_origin = calc_brick_origin(7, (-0.1, -0.1), (0.04, 0.08), 0.015, 0.02)
    fix_brick_id_list = []
    for i in range(len(fix_brick_origin)):
        brick_id = p.loadURDF("meshes/brick/brick_clone.urdf", fix_brick_origin[i], useFixedBase=True) 
        env.object_ids.append(brick_id)
        fix_brick_id_list.append(brick_id)

    brick_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
    brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, brick_orn, useFixedBase=False) 
    env.object_ids.append(brick_id)
    wall_origin = (-0.15, -0.05, 0.1) 
    wall_id = p.loadURDF("meshes/brick/wall.urdf", wall_origin, useFixedBase=True) 
    env.object_ids.append(wall_id)
    
    return brick_id, remove_brick_origin

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

def get_pos_orient():
    return pos_orient   

def user_control_demo(env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler):
    # Touch X Communication
    global msgFromServer
    fname = input("Enter log file name <name>_<date>: ")
    fname += "_manual_2"
    terminate = threading.Event()
    x = threading.Thread(target=getting_phantom_pos, args=(UDPServerSocket, msgFromServer,terminate,))
    x.start()
    adjust_camera(vr_camera_pos, vr_camera_orn_euler)

    state = 0
    count = 0
    grip_attempt = 0
    debug_text = p.addUserDebugText("Ready", (0.2,0.1,0.3), (1,0,0))
    start = time.time()
    z = threading.Thread(target=log_robot_object, args=(env, fname, brick_id, start, terminate, get_pos_orient, remove_brick_origin,))
    z.start()
    duration_text = -1
    break_loop = False
    while True:
        # gaze_pos, pointRay = get_gaze_pos(pointRay)
        
        # pred_class = get_vr_button(env)
        # p.removeUserDebugItem(debug_text)
        events = p.getVREvents()
        for e in events:
            if (e[BUTTONS][7] & p.VR_BUTTON_WAS_TRIGGERED):
                break_loop = True
            if (e[BUTTONS][1] & p.VR_BUTTON_WAS_TRIGGERED):
                warm_up(env)
        if break_loop:
            break
        time_current = time.time()
        p.removeUserDebugItem(duration_text)
        duration = str(round(time_current - start, 2)) + ' s'
        duration_text = p.addUserDebugText(duration, (0.2,0,0.3), (0,1,0))
        slave_pos = list(env.read_debug_parameter()) # get initial position
        x, y, z = rescale_pos_orient(pos_orient)
        joint_pose = env.robot.get_joint_pose()

        slave_pos[0] = slave_pos[0] + x
        slave_pos[1] = slave_pos[1] - y
        slave_pos[2] = z
        # slave_pos[5] = - pos_orient[5] * 0.8
        slave_pos[3] = np.pi/2 - pos_orient[3]
        slave_pos[4] = np.pi/2 + pos_orient[5]
        slave_pos[5] = pos_orient[4]
        if pos_orient[6] == 0:
            slave_pos[6] = 0.085
            with lock:
                msgFromServer[0] = "Push"
            prev_grip = 0
        else:
            slave_pos[6] = 0.0
            with lock:
                msgFromServer[0] = "Push"
            if prev_grip == 0:
                grip_attempt += 1
            prev_grip = 1
        
        slave_pos = tuple(slave_pos)
        obs, reward, done, info = env.step(slave_pos, 'end')

        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)

        d = calculate_distance(np.array(cubePos[:2]), np.array(remove_brick_origin[:2]))
        if d < 0.03 and cubePos[2] < 0.035:
            p.removeUserDebugItem(debug_text)
            debug_text = p.addUserDebugText("Successful", (0.2,0.1,0.3), (0,1,0))
            print(f'Grasping Attempts : {grip_attempt}')
            break
    terminate.set()
    # x.join()
          

if __name__ == '__main__':
    env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler = initialise()
    user_control_demo(env, brick_id, remove_brick_origin, vr_camera_pos, vr_camera_orn_euler)
    env.close()
