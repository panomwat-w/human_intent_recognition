import os

import numpy as np
import pybullet as p

import threading
import json
import socket

import matplotlib.pyplot as plt
from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
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

pos_orient = 0
fname = "dataset/dataset.json"

def getting_phantom_pos(name, msgFromServer):
    global pos_orient
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(address)

        dict_msg = json.loads(message)
        pos_orient = dict_msg["pos_orin"]

        bytesToSend = str.encode(msgFromServer[0])
        UDPServerSocket.sendto(bytesToSend, address)

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
    return label_count

def visualise(motion):
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.plot(motion[:,0], motion[:,1], '-')
    plt.title("movement projection in x-y plane")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.scatter(motion[0,0], motion[0,1], c='r')
    plt.scatter(motion[-1,0], motion[-1,1], c='g')
    plt.xlabel("x")
    plt.ylabel("y")
    

    plt.subplot(1,3,2)
    plt.plot(motion[:,1], motion[:,2], '-')
    plt.title("movement projection in y-z plane")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.scatter(motion[0,1], motion[0,2], c='r')
    plt.scatter(motion[-1,1], motion[-1,2], c='g')
    plt.xlabel("y")
    plt.ylabel("z")


    plt.subplot(1,3,3)
    plt.plot(motion[:,0], motion[:,2], '-')
    plt.scatter(motion[0,0], motion[0,2], c='r')
    plt.scatter(motion[-1,0], motion[-1,2], c='g')
    plt.title("movement projection in x-z plane")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.xlabel("x")
    plt.ylabel("z")

    plt.show()


def record_data():
    global msgFromServer
    calibrate = input("Have you calibrated the haptic device? (y/n) : ").lower()
    if calibrate != "y":
        exit()
    fname = input("Enter file name : ")
    fname = "dataset/" + fname + ".json"
    print("Saving data to : ", fname)
    x = threading.Thread(target=getting_phantom_pos, args=(1,msgFromServer,))
    x.start()
    i = 0
    while type(pos_orient) != list:
        pass
    print("Finish initializing, you can start recording the data now")

    num_samples = int(input("Enter number of samples : "))
    total_classes = int(input("Enter number of classes : "))
    class_current = 0
    motion_list = []
    dataset = []
    prev_pos_orient = pos_orient
    state = 0
    warm_up = 5
    print("#######################")
    # print("Collecting class ", class_current)
    print("Recording warm up data ...")
    print("#######################")
    print("Press button to start recording")
    while True:
        
        if state == 0 and pos_orient[6] == 1:
            state = 1
            print("Button pressed, start recording ...")
            with lock:
                msgFromServer[0] = "Push"

        if state == 1 and pos_orient[6] == 1:
            with lock:
                msgFromServer[0] = "Push"
            # print(pos_orient)
            # if prev_pos_orient != pos_orient:
            #     motion_list.append(pos_orient)
            motion_list.append(pos_orient)
            time.sleep(0.01)

        if state == 1 and pos_orient[6] == 0:
            state = 0
            with lock:
                msgFromServer[0] = "Hello UDP Client"
            print("Button released")
            print("length of motion data : ", len(motion_list))
            for motion in motion_list:
                print(motion)
            
            if warm_up > 0:
                warm_up -= 1
                visualise(np.array(motion_list))
                if warm_up == 0:
                    print("Finish recording warm up data")
                    print("#######################")
                    print("Collecting class ", class_current)
                    print("#######################")
                print("Press button to start recording")
                motion_list = []
                continue


            label = class_current
            # label = (input("Enter label number of this motion : "))
            # while label not in [str(i) for i in range(10)]:
            #     label = (input("Enter label number of this motion : "))
            # label = int(label)
            
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
            
            label_count = report_class(fname)
            if label_count[class_current] >= num_samples:
                print("Class ", class_current, " is completed")
                class_current += 1
                move = input("Do you want to move to next class? (y/n) : ").lower()
                while move != "y":
                    move = input("Do you want to move to next class? (y/n) : ").lower()
            if class_current == total_classes:
                print("Finish collecting data")
                break
            print("#######################")
            print("Collecting class ", class_current)
            print("#######################")
            print("Press button to start recording")
            
            motion_list = []
            # print("Ready for new data")
        prev_pos_orient = pos_orient

if __name__ == '__main__':
    record_data()
