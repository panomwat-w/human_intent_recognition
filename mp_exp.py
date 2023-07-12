import os

import numpy as np
import pybullet as p

import threading
import multiprocessing
import json
import socket

from tqdm import tqdm
from task_environment import *
import time
import math


localIP = "127.0.0.1"
localPort = 12312
bufferSize = 1024
msgFromServer = ["Hello UDP Client"]
lock = threading.Lock()


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

def initialise():
    global msgFromServer
    x = threading.Thread(target=getting_phantom_pos, args=(1,msgFromServer,))
    x.start()

    env = init_env()
    env.reset()
    # env.SIMULATION_STEP_DELAY = 0.

    return env

def remove_all_objects(env):
    for i in range(len(env.object_ids)):
        p.removeBody(env.object_ids[i])
    env.object_ids = []
    env.reset()

def user_control_demo(env):
    global msgFromServer
    model = load_model()
    brick_origin = (-0.3, 0.05, 0.03) 

    fix_brick_origin, remove_brick_origin = calc_brick_origin(7, (-0.1, -0.1), (0.04, 0.08), 0.015, 0.02)
    fix_brick_id_list = []
    for i in range(len(fix_brick_origin)):
        brick_id = p.loadURDF("meshes/brick/brick_clone.urdf", fix_brick_origin[i], useFixedBase=True) 
        env.object_ids.append(brick_id)
        fix_brick_id_list.append(brick_id)

    brick_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
    brick_id = p.loadURDF("meshes/brick/brick.urdf", brick_origin, brick_orn, useFixedBase=False) 
    env.object_ids.append(brick_id)

    env.SIMULATION_STEP_DELAY = 1/100000.0
    warm_up(env)
    env.SIMULATION_STEP_DELAY = 1/240.0

    target_pos=[remove_brick_origin[0], remove_brick_origin[1], 0.28]
    count = 0
    motion_list = []
    prev_pos_orient = pos_orient
    state = 0
    max_len = 170
    while True:
        if state == 0 and pos_orient[6] == 1:
            state = 1
            print("Button pressed, start recording ...")
            with lock:
                msgFromServer[0] = "Push"

        if state == 1 and pos_orient[6] == 1:
            with lock:
                msgFromServer[0] = "Push"
            motion_list.append(pos_orient)
            time.sleep(0.01)

        if state == 1 and pos_orient[6] == 0:
            state = 0
            with lock:
                msgFromServer[0] = "Hello UDP Client"
            print("Button released")
            print("length of motion data : ", len(motion_list))
            if len(motion_list) == 0:
                print("No motion data, please try again")
                motion_list = []
                print("Ready for new data")
                continue
            if len(motion_list) > max_len:
                print("Too much motion data, please try again")
                motion_list = []
                print("Ready for new data")
                continue
            x_star = preprocess(motion_list, max_len)
            pred = model.predict(x_star)
            pred_class = pred[0].argmax()
            print(f"Predicted Class = {pred_class}")
            if pred.max() < 0.6:
                print("Low confidence, please try again")
                motion_list = []
                print("Ready for new data")
                continue
            else:
                print("High confidence, executing motion primitives")
                cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
                obj_pos = [cubePos[0], cubePos[1], cubePos[2]+0.18]
                obs, _, _, _ = apply_motion_primitives(env, pred_class, obj_pos=obj_pos, obj_orn=cubeOrn, des_pos=target_pos)
                cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)

                d = calculate_distance(np.array(cubePos[:2]), np.array(remove_brick_origin[:2]))
                if d < 0.06 and cubePos[2] < 0.035:
                    break
            motion_list = []
            print("Ready for new data")

        if state == 0:
            qKey = ord('q')
            spaceKey = ord('b')
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                break

            if spaceKey in keys and keys[spaceKey] & p.KEY_IS_DOWN:
                with lock:
                    msgFromServer[0] = "Push"
                env.SIMULATION_STEP_DELAY = 0.
                slave_pos = list(env.read_debug_parameter()) # get initial position
                x, y, z = rescale_pos_orient(pos_orient)
                print(np.array(pos_orient).round(3))

                # adjust to the desired position
                slave_pos[0] = slave_pos[0] + x
                slave_pos[1] = slave_pos[1] - y
                slave_pos[2] = z

                slave_pos[5] = - pos_orient[5] * 0.8

                if pos_orient[6] == 0:
                    slave_pos[6] = 0.085
                else:
                    slave_pos[6] = 0.0       

                slave_pos = tuple(slave_pos)
                obs, reward, done, info = env.step(slave_pos, 'end')
                env.SIMULATION_STEP_DELAY = 1/240.
if __name__ == '__main__':
    env = initialise()
    while True:
        user_control_demo(env)
        remove_all_objects(env)