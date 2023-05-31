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
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    motion_data = []
    while True:
        # print(pos_orient_np)
        # print(type(env.read_debug_parameter()))
        # obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')

        z_shit = - 0.2

        slave_pos = list(env.read_debug_parameter()) # get initial position

        # adjust to the desired position
        slave_pos[0] = slave_pos[0] - pos_orient[1]
        slave_pos[1] = slave_pos[1] + pos_orient[0]
        slave_pos[2] = slave_pos[2] + pos_orient[2] + z_shit

        if pos_orient[6] == 0:
            slave_pos[6] = 0.05
        else:
            slave_pos[6] = 0.0
            a = np.stack(motion_data)
            print("Min",a.min(axis=0))
            print("Max",a.max(axis=0))

        # slave_pos.append(0.0)
        print(pos_orient)
        motion_data.append(np.array(pos_orient))
        time.sleep(0.1)
        slave_pos = tuple(slave_pos)

        obs, reward, done, info = env.step(slave_pos, 'end')
        # print(obs, reward, done, info)


if __name__ == '__main__':
    user_control_demo()
