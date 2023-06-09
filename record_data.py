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
fname = "dataset/dataset.json"

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

def report_class(fname):
    data = json.load(open(fname))
    label_count = {}
    for motion in data:
        label = motion["label"]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    print(label_count)


def record_data():
    fname = input("Enter file name : ")
    fname = "dataset/" + fname + ".json"
    # while os.path.isdir(fname):
    #     fname = input("File already exists, enter new file name : ")
    #     fname = "dataset/" + fname + ".json"
    print("Saving data to : ", fname)
    x = threading.Thread(target=getting_phantom_pos, args=(1,))
    x.start()
    i = 0
    while type(pos_orient) != list:
        pass
    print("Finish initializing, you can start recording the data now")

    motion_list = []
    dataset = []
    prev_pos_orient = pos_orient
    state = 0
    while True:
        
        if state == 0 and pos_orient[6] == 1:
            state = 1
            print("Button pressed, start recording ...")

        if state == 1 and pos_orient[6] == 1:
            # print(pos_orient)
            if prev_pos_orient != pos_orient:
                motion_list.append(pos_orient)

        if state == 1 and pos_orient[6] == 0:
            state = 0
            print("Button released")
            print("length of motion data : ", len(motion_list))
            for motion in motion_list:
                print(motion)
            
            label = (input("Enter label number of this motion : "))
            while label not in [str(i) for i in range(10)]:
                label = (input("Enter label number of this motion : "))
            label = int(label)
            
            accept = input("Do you want this data point? (y/n) : ").lower()
            if accept == "y":
                dataset.append(
                    {
                        "label" : label,
                        "motion" : motion_list
                    }
                )
                with open(fname, "w") as f:
                    json.dump(dataset, f, indent=4)
            
            report_class(fname)
            motion_list = []
            print("Ready for new data")
        prev_pos_orient = pos_orient

if __name__ == '__main__':
    record_data()
