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
    model = tf.keras.models.load_model("model/position/best_cnn_model.h5")
    return model

def path_interpolation(start, end, num):
    path = []
    for i in range(num):
        path.append(start + (end - start) / num * i)
    return path

def move_position(env, target_pos, gripper_length, distance=0.05, patience=100):
    slave_pos = list(env.read_debug_parameter()) # get initial position
    slave_pos[0] = target_pos[0]
    slave_pos[1] = target_pos[1]
    slave_pos[2] = target_pos[2]
    slave_pos[6] = gripper_length
    slave_pos = tuple(slave_pos)
    obs, reward, done, info = env.step(slave_pos, 'end')
    d = calculate_distance(obs['ee_pos'], slave_pos[:3])
    i = 0
    while d > distance or i < 10:
        if i > patience:
            break
        obs, reward, done, info = env.step(slave_pos, 'end')
        d = calculate_distance(obs['ee_pos'], slave_pos[:3])
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
        target_pos = np.array([-0.2, -0.2, 0.2])
        gripper_length = 0.04
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
        target_pos = np.array([0.0, 0.0, 0.5])
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
    elif pred_class == 2:
        target_pos = np.array([0.2, -0.2, current_pos[2]])
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)

    elif pred_class == 3:
        target_pos = np.array([0.2, 0.2, current_pos[2]])
        gripper_length = 0.0
        obs, reward, done, info = update_position(env, current_pos, target_pos, gripper_length, num_interpolation=20, interpolation=interpolation)
        current_pos = np.array(obs['ee_pos'])
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

def inference():
    model = load_model()
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

    obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
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
            print(f"Predicted Class = {class_dict[pred_class]}")
            if pred.max() < 0.6:
                print("Low confidence, please try again")
                motion_list = []
                print("Ready for new data")
                continue
            else:
                print("High confidence, executing motion primitives")
                obs, reward, done, info = apply_motion_primitives(env, obs, pred_class, interpolation=False)


            motion_list = []
            print("Ready for new data")
        prev_pos_orient = pos_orient

if __name__ == '__main__':
    inference()
