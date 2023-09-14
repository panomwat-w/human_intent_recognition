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
    p.setRealTimeSimulation(1)
    p.resetSimulation()
    p.setAdditionalSearchPath(project_path)
    
    env = ClutteredPushGraspVR(robot, ycb_models, camera, vis=True)  
    env.reset()
    env.SIMULATION_STEP_DELAY = 1e-2
    vr_camera_pos = [0,-1.9,0.09]
    vr_camera_orn_euler = [-np.pi/2 + np.pi/15,0,0]
    vr_camera_orn_euler = [-np.pi/4,0,0]
    vr_camera_orn = p.getQuaternionFromEuler(vr_camera_orn_euler)
    p.setVRCameraState(vr_camera_pos, vr_camera_orn)
    for i in range(1):
        warm_up(env)
    
    return env, vr_camera_pos, vr_camera_orn_euler



def initialize_brick(env, mp_mode):
    
    fix_brick_origin, remove_brick_origin = calc_brick_origin(5, (0.1, -0.1), (0.04, 0.08), 0.015, 0.02)
    fix_brick_id_list = []
    for i in range(len(fix_brick_origin)):
        brick_id = p.loadURDF("meshes/brick/brick_clone.urdf", fix_brick_origin[i], useFixedBase=True) 
        env.object_ids.append(brick_id)
        fix_brick_id_list.append(brick_id)
    if mp_mode == 0 :
        brick_origin = (-0.18, -0.1, 0.02) 
    else:
        brick_origin = (0.1, -0.25, 0.02) 
    brick_orn = p.getQuaternionFromEuler([0, 0, 0])
    brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, brick_orn, useFixedBase=False) 
    
    p.changeDynamics(brick_id, -1, restitution=0.2, frictionAnchor=1, lateralFriction=2)
    print(p.getDynamicsInfo(brick_id, -1))
    env.object_ids.append(brick_id)
    wall_origin = (0.0, -0.05, 0.1) 
    wall_id = p.loadURDF("meshes/brick/wall.urdf", wall_origin, useFixedBase=True) 
    env.object_ids.append(wall_id)
    
    return brick_id, remove_brick_origin

def initialize_robot(env, brick_id, remove_brick_origin, mp_mode):
    if mp_mode == 0:
        pass
    elif mp_mode == 1:
        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
        obj_pos = [cubePos[0], cubePos[1], cubePos[2]+0.22]
        # obs, reward, done, info= apply_motion_primitives(env, pred_class=3)
        obs, reward, done, info= apply_motion_primitives(env, pred_class=0, obj_pos=obj_pos)
    elif mp_mode == 2:
        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
        obj_pos = [cubePos[0], cubePos[1], cubePos[2]+0.22]
        target_pos=[remove_brick_origin[0], remove_brick_origin[1], 0.28]
        # obs, reward, done, info= apply_motion_primitives(env, pred_class=3)
        obs, reward, done, info= apply_motion_primitives(env, pred_class=0, obj_pos=obj_pos)
        # time.sleep(0.5)
        obs, reward, done, info= apply_motion_primitives(env, pred_class=1, obj_pos=obj_pos, des_pos=target_pos)
    else:
        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
        obj_pos = [cubePos[0], cubePos[1], cubePos[2]+0.22]
        target_pos=[remove_brick_origin[0], remove_brick_origin[1], 0.28]
        # obs, reward, done, info= apply_motion_primitives(env, pred_class=3)
        obs, reward, done, info= apply_motion_primitives(env, pred_class=0, obj_pos=obj_pos)
        # time.sleep(0.5)
        obs, reward, done, info= apply_motion_primitives(env, pred_class=1, obj_pos=obj_pos, des_pos=target_pos)

def apply_motion_primitives(env, pred_class, obj_pos=[-0.2, -0.2, 0.1], obj_orn=[0.0, 0.0, 0.0], des_pos=[0.2, 0.2, 0.1]):
    current_coordinate = env.robot.get_coordinate()
    current_pos = np.array(current_coordinate[0])
    current_length = env.robot.get_gripper_length()
    current_joint_pose = env.robot.get_joint_pose()
    close_length = 0.02
    move_height = 0.6
    if pred_class == 0:
        target_pos = np.array(obj_pos)
        target_pos[2] = move_height
        gripper_length = 0.08
        move_gripper(env, gripper_length)
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pitch=True, target_pitch=np.pi/2, config_yaw=True, target_yaw=np.pi/2, distance=[0.015, 0.015, 0.05])
        current_pos = np.array(obs['ee_pos'])

        target_pos[2] = obj_pos[2]
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_yaw=True, target_yaw=np.pi/2, distance=[0.01, 0.01, 0.01])

        gripper_length = close_length
        move_gripper(env, gripper_length)

        target_pos = current_pos
        target_pos[2] = move_height
        gripper_length = close_length
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pitch=False, target_pitch=np.pi/2, config_yaw=False, target_yaw=None, distance=[0.03, 0.03, 0.05])

    elif pred_class == 1:
        target_pos = np.array(des_pos)
        target_pos[2] = move_height
        gripper_length = close_length
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_yaw=False, target_yaw=None, distance=[0.02, 0.02, 0.05])
    
    elif pred_class == 2:
        target_pos = current_pos
        target_pos[2] = 0.3
        gripper_length = close_length
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pitch=True, target_pitch=np.pi/2, config_yaw=False, distance=[0.03, 0.03, 0.03])
        current_pos = np.array(obs['ee_pos'])

        gripper_length = 0.05
        move_gripper(env, gripper_length)

        target_pos = current_pos
        target_pos[2] = move_height
        gripper_length = 0.08
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_yaw=True, distance=[0.1, 0.1, 0.1])

        target_pos = np.array(env.read_debug_parameter())[:3]
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pitch=True, target_pitch=np.pi/2, config_yaw=True, target_yaw=None, distance=[0.1, 0.1, 0.1])

    elif pred_class == 3:
        target_pos = current_pos
        gripper_length = current_length
        target_joint = list(current_joint_pose.copy())
        increment = np.pi/2
        target_joint[5] = (target_joint[5] + np.pi/2 + increment) % np.pi - np.pi/2
        env.current_yaw = (env.current_yaw + np.pi/2 + increment) % np.pi - np.pi/2
        # debug_text = p.addUserDebugText(str(target_joint[5]), (0.2, 0.1, 0.3))
        obs, reward, done, info = move_joint(env, target_joint, gripper_length)
        # obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pos=False, config_orn=True, target_yaw=target_yaw)
        # p.removeUserDebugItem(debug_text)
    elif pred_class == 4:
        target_pos = current_pos
        gripper_length = current_length
        target_joint = list(current_joint_pose.copy())
        increment = np.pi/2
        target_joint[4] = (target_joint[4] + np.pi + increment) % (2*np.pi) - np.pi
        
        obs, reward, done, info = move_joint(env, target_joint, gripper_length)
        
    else:
        raise ValueError("Invalid class")
    return obs, reward, done, info

def move_position(env, target_pos, gripper_length, config_pos=True, config_yaw=False, config_pitch=False, config_gripper=True, target_yaw=0.0, target_pitch=np.pi/2, distance=[0.01, 0.01, 0.1], patience=100):
    debug_text = p.addUserDebugText("Please Wait", (0.2,0.1,0.3), (1,0,0))
    current_coordinate = env.robot.get_coordinate()
    current_pos = list(current_coordinate[0])
    current_orn = list(p.getEulerFromQuaternion(current_coordinate[1]))
    slave_pos = current_pos + current_orn + [env.robot.get_gripper_length()] # get current position
    if config_pos:
        slave_pos[0] = target_pos[0]
        slave_pos[1] = target_pos[1]
        slave_pos[2] = target_pos[2]
    # slave_pos[3] = 0.0
    if config_pitch:
        slave_pos[4] = target_pitch
    if config_yaw:
        if target_yaw is not None:
            slave_pos[5] = target_yaw
        else:
            slave_pos[5] = env.current_yaw
    #     robot_origin = env.robot.base_pos
    #     target_angle = np.arctan((target_pos[0]-robot_origin[0])/(target_pos[1]-robot_origin[1] + 0.000000001)) * 0.2
    #     target_angle = 0
    #     slave_pos[5] = (env.current_yaw + target_angle + np.pi/2) % np.pi - np.pi/2
    #     print(env.current_yaw, target_angle)
    if config_gripper:
        slave_pos[6] = gripper_length
    slave_pos = tuple(slave_pos)
    obs, reward, done, info = env.step(slave_pos, 'end')
    # d = calculate_distance(current_pos, slave_pos[:3])
    d_x = calculate_distance(current_pos[0], slave_pos[0])
    d_y = calculate_distance(current_pos[1], slave_pos[1])
    d_z = calculate_distance(current_pos[2], slave_pos[2])
    i = 0
    while d_x > distance[0] or d_y > distance[1] or d_z > distance[2] or i < patience:
        if i > patience:
            break
        env.step_simulation()
        # time.sleep(0.001)
        current_coordinate = env.robot.get_coordinate()
        current_pos = list(current_coordinate[0])
        # d = calculate_distance(current_pos, slave_pos[:3])
        d_x = calculate_distance(current_pos[0], slave_pos[0])
        d_y = calculate_distance(current_pos[1], slave_pos[1])
        d_z = calculate_distance(current_pos[2], slave_pos[2])
        print("Pos", i, slave_pos[:3], current_pos, d_x, d_y, d_z)
        i += 1
    obs, reward, done, info = env.step(slave_pos, 'end')
    p.removeUserDebugItem(debug_text)
    return obs, reward, done, info       

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

def check_goal(mp_mode, cubePos, cubeOrn, remove_brick_origin, init_cubeOrn=None):
    if mp_mode == 0:
        if cubePos[2] > 0.3 :
            return True
        else :
            return False
    elif mp_mode == 1:
        d = calculate_distance(np.array(cubePos[:2]), np.array(remove_brick_origin[:2]))
        if d < 0.03:
            return True
        else:
            return False
    elif mp_mode == 2:
        d = calculate_distance(np.array(cubePos[:2]), np.array(remove_brick_origin[:2]))
        if d < 0.03 and cubePos[2] < 0.025:
            return True
        else:
            return False
    elif mp_mode == 3:
        if np.abs(cubeOrn[2] - init_cubeOrn[2]) >= np.pi/2:
            return True
        else:
            return False

def get_pos_orient():
    return pos_orient   

def user_control_demo(env, vr_camera_pos, vr_camera_orn_euler, participant_name, mp_mode=1):
    # Touch X Communication
    global msgFromServer
    brick_id, remove_brick_origin = initialize_brick(env, mp_mode)
    time.sleep(1)
    for i in range(1):
        warm_up(env)
    initialize_robot(env, brick_id, remove_brick_origin, 0)
    fname = participant_name + f"_manual_1_{mp_mode}"
    terminate = threading.Event()
    x = threading.Thread(target=getting_phantom_pos, args=(UDPServerSocket, msgFromServer,terminate,))
    x.start()
    adjust_camera(vr_camera_pos, vr_camera_orn_euler)

    state = 0
    count = 0
    debug_text = p.addUserDebugText("Ready", (0.2,0.1,0.3), (1,0,0))
    
    duration_text = -1
    prev_grip = 0
    grip_attempt = 0
    timing_start = False
    init_cubeOrn = None
    while True:
        # gaze_pos, pointRay = get_gaze_pos(pointRay)
        
        # pred_class = get_vr_button(env)
        # p.removeUserDebugItem(debug_text)
        break_loop = False
        if timing_start:
            time_current = time.time()
            p.removeUserDebugItem(duration_text)
            duration = str(round(time_current - start, 2)) + ' s'
            duration_text = p.addUserDebugText(duration, (0.2,0,0.3), (0,1,0))
            events = p.getVREvents()
            for e in events:
                if (e[BUTTONS][7] & p.VR_BUTTON_WAS_TRIGGERED):
                    break_loop= True
                if (e[BUTTONS][1] == p.VR_BUTTON_IS_DOWN):
                    warm_up(env)
        else:
            events = p.getVREvents()
            for e in events:
                if (e[BUTTONS][33] & p.VR_BUTTON_WAS_TRIGGERED):
                    start = time.time()
                    z = threading.Thread(target=log_robot_object, args=(env, fname, brick_id, start, terminate, get_pos_orient, remove_brick_origin,))
                    z.start()
                    timing_start = True
                    if mp_mode == 3:
                        init_cubePos, init_cubeOrn = p.getBasePositionAndOrientation(brick_id)
                        init_cubeOrn = p.getEulerFromQuaternion(init_cubeOrn)
                if (e[BUTTONS][1] == p.VR_BUTTON_IS_DOWN):
                    warm_up(env)
                if (e[BUTTONS][7] & p.VR_BUTTON_WAS_TRIGGERED):
                    break_loop= True
                    terminate.set()
        if break_loop :
                break

        slave_pos = list(env.read_debug_parameter()) # get initial position
        x, y, z = rescale_pos_orient(pos_orient)

        slave_pos[0] = slave_pos[0] + x
        slave_pos[1] = slave_pos[1] - y
        slave_pos[2] = z

        # slave_pos[5] = - pos_orient[5] * 0.8
        slave_pos[3] = np.pi/2 - pos_orient[3]
        slave_pos[4] = np.pi/2 - pos_orient[5]
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


        if timing_start:

            cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
            cubeOrn = p.getEulerFromQuaternion(cubeOrn)
            if mp_mode == 3:
                print(cubeOrn[2], init_cubeOrn[2], cubeOrn[2]-init_cubeOrn[2])

            if check_goal(mp_mode, cubePos, cubeOrn, remove_brick_origin, init_cubeOrn):
                completion_time = time.time() - start
                p.removeUserDebugItem(debug_text)
                p.removeUserDebugItem(duration_text)
                debug_text = p.addUserDebugText(f"Successful in {round(completion_time, 2)} sec", (0.2,0.1,0.3), (0,1,0))
                grasping_text = -1
                if mp_mode == 0:
                    grasping_text = p.addUserDebugText(f"Grasping attempts : {grip_attempt} times", (0.2,0,0.3), (0,1,0))
                    print(f'Grasping Attempts : {grip_attempt}')
                terminate.set()
                time.sleep(2)
                p.removeUserDebugItem(debug_text)
                p.removeUserDebugItem(grasping_text)
                break
    terminate.set()
    # x.join()
          

if __name__ == '__main__':
    env, vr_camera_pos, vr_camera_orn_euler = initialise()
    participant_name = input("Enter participant name : ")
    while True:
        while True:
            try:
                mp_mode = input("Enter manual sub 1 mode : ")
                mp_mode = int(mp_mode)
                if input(f"Confirm manual sub 1 mode {mp_mode} (y/n) : ") == 'y':
                    break
                
            except:
                print('Input incorrect')
        user_control_demo(env, vr_camera_pos, vr_camera_orn_euler, participant_name, mp_mode)
        time.sleep(2)
        remove_all_objects(env)
        if input(f"Exit? (y/n) : ") == 'y':
            break
    env.close()
