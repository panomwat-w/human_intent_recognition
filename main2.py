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

def rescale_pos_orient(pos_orient):
    x_range_device = [-0.21, 0.2] # mid val = -0.16 range = 0.14
    y_range_device = [-0.15, 0.15] # mid val = -0.07 range = 0.38
    z_range_device = [-0.11, 0.144] # mid val = 0.31 range = 0.31

    x_range = [-0.224, 0.224]
    y_range = [-0.224, 0.224]
    z_range = [0.0, 1.0]

    x = ((pos_orient[1] - x_range_device[0])/(x_range_device[1] - x_range_device[0]) - 0.5)*0.8
    y = ((pos_orient[0] - y_range_device[0])/(y_range_device[1] - y_range_device[0]) - 0.5)*0.8
    z = ((pos_orient[2] - z_range_device[0])/(z_range_device[1] - z_range_device[0]))*0.7
    # z = pos_orient[2] - 0.2

    return x, y, z


def user_control_demo():
    x = threading.Thread(target=getting_phantom_pos, args=(1,))
    x.start()

    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()

    # env.SIMULATION_STEP_DELAY = 0
    while True:
        slave_pos = list(env.read_debug_parameter()) # get initial position
        x, y, z = rescale_pos_orient(pos_orient)

        # adjust to the desired position
        slave_pos[0] = slave_pos[0] + x
        slave_pos[1] = slave_pos[1] - y
        slave_pos[2] = z
        slave_pos[5] = -pos_orient[4]/2.5

        print(np.array(slave_pos[5]).round(2))

        if pos_orient[6] == 0:
            slave_pos[6] = 0.085
        else:
            slave_pos[6] = 0.0

        slave_pos = tuple(slave_pos)
        obs, reward, done, info = env.step(slave_pos, 'end')

if __name__ == '__main__':
    user_control_demo()
