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
        # print(dict_msg)
        pos_orient = dict_msg["pos_orin"]

        # print(clientMsg)
        # print(clientIP)
        # print(pos_orient_np.size)

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

def move_gripper(env, gripper_length):
    for i in range(100):
        env.robot.move_gripper(gripper_length)
        env.step_simulation()

# def update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=False):
#     if interpolation:
#         path = path_interpolation(current_pos, target_pos, num_interpolation)
#         for i in range(len(path)):
#             obs, reward, done, info= move_position(env, path[i], gripper_length)
#     else:
#         obs, reward, done, info = move_position(env, target_pos, gripper_length)
#     return obs, reward, done, info


def calculate_distance(target, current):
    return np.linalg.norm(np.array(current) - np.array(target))

def warm_up(env):
    print('warm_up')
    x = np.linspace(-0.2, 0.2, 3)
    y = np.linspace(-0.2, 0.2, 3)
    for i in range(len(x)):
        target_pos = (x[i], 0, 0.5)
        obs, reward, done, info = move_position(env, target_pos, 0.085)
    # for i in range(len(y)):
    #     target_pos = (0, y[i], 0.5)
    #     obs, reward, done, info = move_position(env, target_pos, 0.085)
    move_position(env, np.array([0.0,0.0,0.5]), 0.085)
    return obs, reward, done, info

def apply_motion_primitives(env, obs, pred_class, obj_pos=[-0.2, -0.2, 0.1], des_pos=[0.2, 0.2, 0.1]):
    print("pred_class: ", pred_class)
    current_pos = np.array(obs['ee_pos'])
    if pred_class == 0:
        target_pos = np.array(obj_pos)
        gripper_length = 0.085
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
        current_pos = np.array(obs['ee_pos'])

        gripper_length = 0.0
        move_gripper(env, gripper_length)

    elif pred_class == 1:
        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.0
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
    
    elif pred_class == 2:
        target_pos = np.array(des_pos)
        target_pos[2] = 0.5
        gripper_length = 0.0
        obs, reward, done, info = move_position(env, target_pos, gripper_length)

    elif pred_class == 3:
        target_pos = np.array(des_pos)
        gripper_length = 0.0
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
        current_pos = np.array(obs['ee_pos'])

        gripper_length = 0.085
        move_gripper(env, gripper_length)

        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.085
        obs, reward, done, info = move_position(env, target_pos, gripper_length)

        target_pos = np.array(env.read_debug_parameter())[:3]
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
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
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()
    brick_origin = (-0.2, -0.2, 0.0) 
    brick_id_list = []
    for i in range(4):
        brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, useFixedBase=False) 
        brick_id_list.append(brick_id)
        brick_origin = (brick_origin[0]+0.15, brick_origin[1], brick_origin[2])
    print(brick_id_list)
    base_id = p.loadURDF("meshes/brick/base.urdf", [0.0,0.0,0.0], useFixedBase=True) 

    env.SIMULATION_STEP_DELAY = 1/100000.0
    obs, reward, done, info= warm_up(env)
    env.SIMULATION_STEP_DELAY = 1/240.0
    j = 0
    target_pos=[0.2, 0.2, 0.23]
    count = 0
    while j < len(brick_id_list):
        # pred_class = int(input("Enter predicted class : "))
        pred_class = count % 4
        count += 1
        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id_list[j])
        print("cubePos before: ", cubePos)
        obj_pos = [cubePos[0], cubePos[1], cubePos[2]+0.2]
        obs, reward, done, info= apply_motion_primitives(env, obs, pred_class, obj_pos=obj_pos, des_pos=target_pos)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id_list[j])
        print("cubePos after: ", cubePos)
        if cubePos[0] > 0.1 and cubePos[1] > 0.1 and pred_class == 3:
            # brick_id_list.pop(0)
            print("brick_id_list: ", brick_id_list)
            # p.removeBody(brick_id_list[0])
            target_pos[2] = cubePos[2] + 0.2
            j += 1

if __name__ == '__main__':
    user_control_demo()
