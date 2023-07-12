import os

import numpy as np
import pybullet as p
import pandas as pd

import threading
import multiprocessing
import json
import socket

from tqdm import tqdm
from task_environment import stream_joint_pose, init_env, log_robot_object
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
from task_environment import calc_brick_origin
import time
import math
import csv
###################
# print('\n==============\nRobot Master Client\n==============\n')
# print('\n')

# ###################

# ip_address = '127.0.0.1'
# port = '11001'

# # Create a TCP/IP socket
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Bind the socket to the port
# server_address = (ip_address, int(port))
# print('connect to server at %s port %s' % server_address)
# s.connect(server_address)

# s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 100)

# # Enum of message types, must match InterboClient
# class MessageType:
#     Invalid, Acknowledge, Goodbye, PoseUpdate = range(4)

# # Example message: 
# # update_pose -180,30,75,-10,90,0,1

# # Key Parameters
# default_buffer_size = 1024
# buffer_size = default_buffer_size

###################

# localIP = "127.0.0.1"
# localPort = 12312
# bufferSize = 1024
# msgFromServer = "Hello UDP Client"
# bytesToSend = str.encode(msgFromServer)


# # Create a datagram socket
# UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# # Bind to address and ip
# UDPServerSocket.bind((localIP, localPort))
# print("UDP server up and listening")

# pos_orient = 0
# silhouette_id = None
# object_position = ()

# def getting_phantom_pos(name):
#     global pos_orient
#     while (True):
#         bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
#         message = bytesAddressPair[0]
#         address = bytesAddressPair[1]

#         clientMsg = "Message from Client:{}".format(message)
#         clientIP = "Client IP Address:{}".format(address)

#         dict_msg = json.loads(message)
#         # print(dict_msg)
#         pos_orient = dict_msg["pos_orin"]

#         # print(clientMsg)
#         # print(clientIP)
#         # print(pos_orient_np.size)

def path_interpolation(start, end, num):
    path = []
    for i in range(num):
        path.append(start + (end - start) / num * i)
    return path

def calc_joint_pos(slave_pos, env):
    x, y, z, roll, pitch, yaw = slave_pos
    pos = (x, y, z)
    orn = p.getQuaternionFromEuler((roll, pitch, yaw))
    joint_poses = list(p.calculateInverseKinematics(env.robot.id, env.robot.eef_id, pos, orn,
                                                env.robot.arm_lower_limits, env.robot.arm_upper_limits, env.robot.arm_joint_ranges, env.robot.arm_rest_poses,
                                                maxNumIterations=20))
    joint_poses[6] = np.pi/2
    return joint_poses

def move_position(env, target_pos, gripper_length, distance=0.005, patience=100):
    # print('move_position')
    slave_pos = list(env.read_debug_parameter()) # get initial position
    slave_pos[0] = target_pos[0]
    slave_pos[1] = target_pos[1]
    slave_pos[2] = target_pos[2]
    slave_pos[6] = gripper_length
    slave_pos = tuple(slave_pos)
    obs, reward, done, info = env.step(slave_pos, 'end')
    d = calculate_distance(obs['ee_pos'], slave_pos[:3])
    i = 0
    while d > distance:
        if i > patience:
            break
        env.step_simulation()
        d = calculate_distance(obs['ee_pos'], slave_pos[:3])
        i += 1
    obs, reward, done, info = env.step(slave_pos, 'end')
    return obs, reward, done, info

def move_gripper(env, gripper_length, distance=0.01, patience=150):
    env.robot.move_gripper(gripper_length)
    current_length = env.robot.get_gripper_length()
    d = abs(current_length - gripper_length)
    i = 0
    while d > distance and i < patience:
        env.step_simulation()
        i += 1

def calculate_distance(target, current):
    return np.linalg.norm(np.array(current) - np.array(target))

def warm_up(env):
    print('warm_up')
    x = np.linspace(-0.2, 0.2, 3)
    y = np.linspace(-0.2, 0.2, 3)
    for i in range(len(x)):
        target_pos = (x[i], 0, 0.5)
        obs, reward, done, info = move_position(env, target_pos, 0.085)
    move_position(env, np.array([0.0,0.0,0.5]), 0.085)
    return obs, reward, done, info

def user_control_demo():
    env = init_env()
    env.reset()
    brick_origin = (-0.2, -0.2, 0.0) 
    brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, useFixedBase=True) 
    df = pd.read_csv('logs/robot_object.csv')
    # df.drop_duplicates(inplace=True)
    print(df.shape)
    # for i in range(df.shape[0]):
    #     robot_joint = df.iloc[i, 0:6].values
    #     gripper = df.iloc[i, 6]
    #     env.robot.move_ee(robot_joint, control_method='joint')
    #     env.robot.move_gripper(gripper)
    #     object_position = df.iloc[i, -7:-4].values
    #     object_orientation = df.iloc[i, -4:].values
    #     # print(object_position, object_orientation)
    #     p.removeBody(brick_id)
    #     brick_id = p.loadURDF("meshes/brick/brick.urdf", object_position, object_orientation, useFixedBase=True)
    #     p.stepSimulation()
    #     # time.sleep(0.00001)
            
    # while True:
    #     p.removeBody(brick_id)
    #     cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
    #     brick_id = p.loadURDF("meshes/brick/brick.urdf", [cubePos[0], cubePos[1], 0.0], cubeOrn, useFixedBase=True)
    #     time.sleep(0.001)

if __name__ == '__main__':
    user_control_demo()
