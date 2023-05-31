import os

import numpy as np
import pybullet as p

import threading
import json
import socket

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math


localIP = "127.0.0.1"
localPort = 12312
bufferSize = 1024
msgFromServer = "Hello UDP Client"
bytesToSend = str.encode(msgFromServer)


# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")

pos_orient = 0

def getting_phantom_pos(name):
    global pos_orient
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(address)

        dict_msg = json.loads(message)
        print(dict_msg)
        pos_orient = dict_msg["pos_orin"]

        # print(clientMsg)
        # print(clientIP)
        # print(pos_orient_np.size)

def path_interpolation(start, end, num):
    path = []
    for i in range(num):
        path.append(start + (end - start) / num * i)
    return path

def move_position(env, target_pos, gripper_length, distance=0.05, patience=100):
    print('move_position')
    slave_pos = list(env.read_debug_parameter()) # get initial position
    slave_pos[0] = target_pos[0]
    slave_pos[1] = target_pos[1]
    slave_pos[2] = target_pos[2]
    slave_pos[6] = gripper_length
    slave_pos = tuple(slave_pos)
    obs, reward, done, info = env.step(slave_pos, 'end')
    d = calculate_distance(obs['ee_pos'], slave_pos[:3])
    i = 0
    while d > distance or i < 50:
        if i > patience:
            break
        obs, reward, done, info = env.step(slave_pos, 'end')
        d = calculate_distance(obs['ee_pos'], slave_pos[:3])
        print(obs['ee_pos'])
        i += 1
    return obs, reward, done, info

def update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=False):
    if interpolation:
        path = path_interpolation(current_pos, target_pos, num_interpolation)
        for i in range(len(path)):
            obs, reward, done, info= move_position(env, path[i], gripper_length)
    else:
        obs, reward, done, info= move_position(env, target_pos, gripper_length)
    return obs, reward, done, info


def calculate_distance(target, current):
    return np.linalg.norm(np.array(current) - np.array(target))

def apply_motion_primitives(env, obs, pred_class, interpolation=False):
    print("pred_class: ", pred_class)
    current_pos = np.array(obs['ee_pos'])
    if pred_class == 0:
        target_pos = np.array([-0.2, -0.2, 0.2])
        gripper_length = 0.04
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])

    elif pred_class == 1:
        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
    elif pred_class == 2:
        target_pos = np.array([0.2, 0.2, current_pos[2]])
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)

    elif pred_class == 3:
        target_pos = current_pos
        target_pos[2] = 0.2
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
        gripper_length = 0.1
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.085
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        target_pos = np.array(env.read_debug_parameter())[:3]
        gripper_length = 0.085
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
    else:
        raise ValueError("Invalid class")
    
    return obs, reward, done, info

def user_control_demo():
    # x = threading.Thread(target=getting_phantom_pos, args=(1,))
    # x.start()

    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()

    obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
    # env.SIMULATION_STEP_DELAY = 0
    while True:

        z_shit = - 0.2

        slave_pos = list(env.read_debug_parameter()) # get initial position

        # adjust to the desired position
        # slave_pos[0] = slave_pos[0] - pos_orient[1]
        # slave_pos[1] = slave_pos[1] + pos_orient[0]
        # slave_pos[2] = slave_pos[2] + pos_orient[2] + z_shit

        # if pos_orient[6] == 0:
        #     slave_pos[6] = 0.05
        # else:
        #     slave_pos[6] = 0.0

        # target_pos = np.array([float(pos) for pos in input('Enter target position x,y,z: ').split()])
        pred_class = int(input("Enter predicted class : "))
        obs, reward, done, info= apply_motion_primitives(env, obs, pred_class, interpolation=False)



if __name__ == '__main__':
    user_control_demo()
