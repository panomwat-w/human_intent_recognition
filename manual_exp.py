import os

import numpy as np
import pybullet as p

import threading
import json
import socket

from tqdm import tqdm
from env import ClutteredPushGrasp
from task_environment import stream_joint_pose, init_env, log_robot_object, calc_brick_origin, calculate_distance
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math


localIP = "127.0.0.1"
localPort = 12312
bufferSize = 1024
msgFromServer = ["Hello UDP Client"]
lock = threading.Lock()
# bytesToSend = str.encode(msgFromServer)


# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")

pos_orient = [0]*7

def getting_phantom_pos(name, msgFromServer):
    global pos_orient
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(address)

        dict_msg = json.loads(message)
        # print(dict_msg)
        pos_orient = dict_msg["pos_orin"]

        # MESSAGE = str.encode("Hi there UDP Client")
        # UDPServerSocket.sendto(MESSAGE, (localIP, localPort))
        # msgFromServer = "Hello UDP Client"
        bytesToSend = str.encode(msgFromServer[0])
        UDPServerSocket.sendto(bytesToSend, address)

        # print(clientMsg)
        # print(clientIP)
        # print(pos_orient_np.size)

def rescale_pos_orient(pos_orient):
    x_range_device = [-0.21, 0.2] # mid val = -0.16 range = 0.14
    y_range_device = [-0.15, 0.15] # mid val = -0.07 range = 0.38
    z_range_device = [-0.11, 0.144] # mid val = 0.31 range = 0.31

    x_range = [-0.224, 0.224]
    y_range = [-0.224, 0.224]
    z_range = [0.0, 1.0]

    x = ((pos_orient[1] - x_range_device[0])/(x_range_device[1] - x_range_device[0]) - 0.5)*0.85
    y = ((pos_orient[0] - y_range_device[0])/(y_range_device[1] - y_range_device[0]) - 0.5)*0.85
    z = ((pos_orient[2] - z_range_device[0])/(z_range_device[1] - z_range_device[0]))*0.7
    # z = pos_orient[2] - 0.2

    return x, y, z


def initialise():
    global msgFromServer
    x = threading.Thread(target=getting_phantom_pos, args=(1,msgFromServer,))
    x.start()

    env = init_env()
    env.reset()
    env.SIMULATION_STEP_DELAY = 0.

    return env

def remove_all_objects(env):
    for i in range(len(env.object_ids)):
        p.removeBody(env.object_ids[i])
    env.object_ids = []
    env.reset()


    

def user_control_demo(env):
    global msgFromServer
    
    # x = threading.Thread(target=getting_phantom_pos, args=(1,msgFromServer,))
    # x.start()

    # env = init_env()
    # env.reset()
    # env.SIMULATION_STEP_DELAY = 0.

    brick_origin = (-0.25, 0.1, 0.03) 

    fix_brick_origin, remove_brick_origin = calc_brick_origin(4, (-0.1, -0.1), (0.06, 0.12), 0.02, 0.03)
    fix_brick_id_list = []
    for i in range(len(fix_brick_origin)):
        brick_id = p.loadURDF("meshes/brick/brick_clone.urdf", fix_brick_origin[i], useFixedBase=True) 
        env.object_ids.append(brick_id)
        fix_brick_id_list.append(brick_id)

    brick_orn = p.getQuaternionFromEuler([0, 0, np.pi/4])
    brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, brick_orn, useFixedBase=False) 
    env.object_ids.append(brick_id)

    # env.SIMULATION_STEP_DELAY = 0
    while True:
        slave_pos = list(env.read_debug_parameter()) # get initial position
        x, y, z = rescale_pos_orient(pos_orient)
        joint_pose = env.robot.get_joint_pose()
        print(np.array(joint_pose).round(2)[5], np.array(pos_orient).round(2)[5])

        # adjust to the desired position
        slave_pos[0] = slave_pos[0] + x
        slave_pos[1] = slave_pos[1] - y
        slave_pos[2] = z
        # slave_pos[4] = np.pi/2 + pos_orient[3]*0.4
        # slave_pos[3] = pos_orient[5]
        # slave_pos[4] = np.pi/2 + pos_orient[3]
        # slave_pos[5] = pos_orient[4]
        # slave_pos[3] = pos_orient[3]
        # slave_pos[4] = np.pi/2 + pos_orient[4]
        slave_pos[5] = - pos_orient[5] * 0.8
        print(round(z, 3))

        # print(np.array(pos_orient).round(2))

        if pos_orient[6] == 0:
            slave_pos[6] = 0.085
            with lock:
                msgFromServer[0] = "Push"
        else:
            slave_pos[6] = 0.0
            with lock:
                msgFromServer[0] = "Push"
    

        slave_pos = tuple(slave_pos)
        obs, reward, done, info = env.step(slave_pos, 'end')

        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
        d = calculate_distance(np.array(cubePos[:2]), np.array(remove_brick_origin[:2]))
        # print(np.array(cubePos).round(2), np.array(remove_brick_origin).round(2), d)
        if d < 0.06 and cubePos[2] < 0.035:
            break

        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
            break
    print('Successful')

if __name__ == '__main__':
    env = initialise()
    while True:
        user_control_demo(env)
        remove_all_objects(env)
