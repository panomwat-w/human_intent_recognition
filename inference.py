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
import tensorflow as tf
import matplotlib.pyplot as plt

max_len = 44
localIP = "127.0.0.1"
localPort = 12312
bufferSize = 1024
msgFromServer = "Hello UDP Client"
bytesToSend = str.encode(msgFromServer)
class_dict = {0:"Class 0: Grasping", 1:"Class 1: Lifting", 2:"Class 2: Moving", 3:"Class 3: Placing"}


# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")

pos_orient = 0
# fname = "dataset/dataset.json"


def preprocess(motion_list, max_len):
    tmp_x = np.array(motion_list)[:,:3] # exclude the last dimension corresponding to button pressing
    tmp_x = np.concatenate([tmp_x, np.zeros((max_len - tmp_x.shape[0], tmp_x.shape[1]))]) # zero padding
    # tmp_x_lag = np.concatenate([np.zeros((1, tmp_x.shape[1])), tmp_x[:-1, :]]) # lag 1
    # velocity = tmp_x - tmp_x_lag
    # tmp_x = np.concatenate([tmp_x, velocity], axis=1)
    tmp_x = np.expand_dims(tmp_x, axis=0)
    return tmp_x

def getting_phantom_pos(name):
    global pos_orient
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(address)

        dict_msg = json.loads(message)
        pos_orient = dict_msg["pos_orin"]

        # print(clientMsg)
        # print(clientIP)
        # print(pos_orient_np.size)


def load_model():
    model = tf.keras.models.load_model("best_cnn_model.h5")
    return model

def inference():
    model = load_model()
    x = threading.Thread(target=getting_phantom_pos, args=(1,))
    x.start()
    i = 0
    while type(pos_orient) != list:
        pass
    print("Finish initializing")

    motion_list = []
    prev_pos_orient = pos_orient
    state = 0
    while True:
        
        if state == 0 and pos_orient[6] == 1:
            state = 1
            print("Button pressed, start recording ...")

        if state == 1 and pos_orient[6] == 1:
            if prev_pos_orient != pos_orient:
                motion_list.append(pos_orient)

        if state == 1 and pos_orient[6] == 0:
            state = 0
            print("Button released")
            print("length of motion data : ", len(motion_list))
            motion_array = np.array(motion_list)
            # plt.plot(motion_array[:,0], motion_array[:,1], 'o-')
            # plt.show()
            plt.figure(figsize=(5,5))
            ax = plt.axes(projection='3d')
            ax.plot3D(motion_array[:,0], motion_array[:,1], motion_array[:,2])
            plt.xlabel("x")
            plt.ylabel("y")
            ax.set_zlabel("z")
            plt.xlim(-0.2, 0.2)
            plt.ylim(-0.2, 0.2)
            ax.set_zlim(-0.2, 0.2)
            plt.show()

            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.plot(motion_array[:,0], motion_array[:,1], 'o-')
            plt.xlim(-0.2, 0.2)
            plt.ylim(-0.2, 0.2)
            plt.title("x-y")
            plt.subplot(1,3,2)
            plt.plot(motion_array[:,1], motion_array[:,2], 'o-')
            plt.xlim(-0.2, 0.2)
            plt.ylim(-0.2, 0.2)
            plt.title("y-z")
            plt.subplot(1,3,3)
            plt.plot(motion_array[:,0], motion_array[:,2], 'o-')
            plt.xlim(-0.2, 0.2)
            plt.ylim(-0.2, 0.2)
            plt.title("x-z")
            plt.show()

            x_star = preprocess(motion_list, max_len)
            pred = model.predict(x_star)
            pred_class = pred[0].argmax()
            print(f"Predicted Class = {class_dict[pred_class]}")



            motion_list = []
            print("Ready for new data")
        prev_pos_orient = pos_orient

if __name__ == '__main__':
    inference()
