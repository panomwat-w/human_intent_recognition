import os
import time
import numpy as np
import pybullet as p
import tensorflow as tf

from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera

import threading
import json
import socket

# from pybullet_ur5_robotiq.envs import ClutteredPushGrasp
# from pybullet_ur5_robotiq.envs.camera import Camera
# from pybullet_ur5_robotiq.envs.robot import UR5Robotiq140
# from pybullet_ur5_robotiq.envs.ycb import YCBModels


def init_env():
    ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'),)
    camera = Camera((0.12, -0.1, 1.5),
                    (0.12, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (160, 160), 40)
    camera = None
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    return env

def calc_brick_origin(max_num_bricks=5, brick_origin=(0.0,0.0), brick_dim=(0.06,0.12,0.06), margin=0.006, z=0.03):
    fix_brick_origin_list = []
    for i in range(1, max_num_bricks + 1):
        if i % 2 != 0:
            fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1], z))
            for j in range(1, int((i-1)/2) + 1):
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]+(brick_dim[1]+margin)*j, z))
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]-(brick_dim[1]+margin)*j, z))
            if i < max_num_bricks:
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1], z))
                for j in range(1, int((i-1)/2) + 1):
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]+(brick_dim[1]+margin)*j, z))
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]-(brick_dim[1]+margin)*j, z))
        else:
            for j in range(1, int(i/2) + 1):
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]+(brick_dim[1]+margin)*(j-0.5), z))
                fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(i-1), brick_origin[1]-(brick_dim[1]+margin)*(j-0.5), z))
                if i < max_num_bricks:
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]+(brick_dim[1]+margin)*(j-0.5), z))
                    fix_brick_origin_list.append((brick_origin[0]+(brick_dim[0]+margin)*(2 * max_num_bricks - i - 1), brick_origin[1]-(brick_dim[1]+margin)*(j-0.5), z))
    
    random_idx = np.random.randint(0, len(fix_brick_origin_list))
    remove_brick_origin = fix_brick_origin_list.pop(random_idx)
    return fix_brick_origin_list, remove_brick_origin

        
def move_position(env, target_pos, gripper_length, config_pos=True, config_yaw=False, config_pitch=False, config_gripper=True, target_yaw=0.0, target_pitch=np.pi/2, distance=0.005, patience=100):
    current_coordinate = env.robot.get_coordinate()
    current_pos = list(current_coordinate[0])
    current_orn = list(p.getEulerFromQuaternion(current_coordinate[1]))
    slave_pos = current_pos + current_orn + [env.robot.get_gripper_length()] # get current position
    if config_pos:
        slave_pos[0] = target_pos[0]
        slave_pos[1] = target_pos[1]
        slave_pos[2] = target_pos[2]
    # slave_pos[3] = 0.0
    if config_pitch:
        slave_pos[4] = target_pitch
    if config_yaw:
        slave_pos[5] = target_yaw
    if config_gripper:
        slave_pos[6] = gripper_length
    slave_pos = tuple(slave_pos)
    current_yaw = p.getEulerFromQuaternion(env.robot.get_coordinate()[1])[2]
    obs, reward, done, info = env.step(slave_pos, 'end')
    d = calculate_distance(obs['ee_pos'], slave_pos[:3])
    d_yaw = abs((current_yaw - target_yaw) % np.pi)
    i = 0
    while d > distance and d_yaw > distance:
        if i > patience:
            break
        env.step_simulation()
        d = calculate_distance(obs['ee_pos'], slave_pos[:3])
        current_yaw = p.getEulerFromQuaternion(env.robot.get_coordinate()[1])[2]
        d_yaw = abs((current_yaw - target_yaw) % np.pi)
        i += 1
    obs, reward, done, info = env.step(slave_pos, 'end')
    return obs, reward, done, info

def move_joint(env, target_joint, gripper_length, distance=0.01, patience=150):
    slave_pos = list(env.read_debug_parameter()) # get initial position
    slave_pos[0] = target_joint[0]
    slave_pos[1] = target_joint[1]
    slave_pos[2] = target_joint[2]
    slave_pos[3] = target_joint[3]
    slave_pos[4] = target_joint[4]
    slave_pos[5] = target_joint[5]
    slave_pos[6] = gripper_length
    slave_pos = tuple(slave_pos)
    env.step(slave_pos, 'joint')
    current_joint = env.robot.get_joint_pose()
    # d = calculate_distance(current_joint, target_joint)
    d = abs(target_joint[5] - current_joint[5])

    i = 0
    while d > distance:
        if i > patience:
            break
        env.step_simulation()
        # d = calculate_distance(current_joint, target_joint)
        current_joint = env.robot.get_joint_pose()
        d = abs(target_joint[5] - current_joint[5])
        print("Yaw", i, target_joint[5], current_joint[5], d)
        i += 1
    obs, reward, done, info = env.step(slave_pos, 'joint')
    return obs, reward, done, info

def move_gripper(env, gripper_length, distance=0.01, patience=300):
    # env.SIMULATION_STEP_DELAY = 0.0
    env.robot.move_gripper(gripper_length)
    current_length = env.robot.get_gripper_length()
    d = abs(current_length - gripper_length)
    i = 0
    while d > distance and i < patience:
        env.robot.move_gripper(gripper_length)
        env.step_simulation()
        current_length = env.robot.get_gripper_length()
        d = abs(current_length - gripper_length)
        print("Grip", i, current_length, gripper_length, d)
        i += 1
    # env.SIMULATION_STEP_DELAY = 1 / 240.

def calculate_distance(target, current):
    return np.linalg.norm(np.array(current) - np.array(target))

def warm_up(env):
    print('warm_up')
    # x = np.linspace(-0.2, 0.2, 3)
    # y = np.linspace(-0.2, 0.2, 3)
    # for i in range(len(x)):
    #     target_pos = (x[i], 0, 0.5)
    #     obs, reward, done, info = move_position(env, target_pos, 0.085)
    obs, reward, done, info = move_position(env, np.array([0.0,0.0,0.5]), 0.08)
    return obs, reward, done, info


def apply_motion_primitives(env, pred_class, obj_pos=[-0.2, -0.2, 0.1], obj_orn=[0.0, 0.0, 0.0], des_pos=[0.2, 0.2, 0.1]):
    print("pred_class: ", pred_class)
    current_coordinate = env.robot.get_coordinate()
    current_pos = np.array(current_coordinate[0])
    current_orn = np.array(p.getEulerFromQuaternion(current_coordinate[1]))
    # current_pos = np.array(obs['ee_pos'])
    current_length = env.robot.get_gripper_length()
    current_joint_pose = env.robot.get_joint_pose()
    if pred_class == 0:
        target_pos = np.array(obj_pos)
        target_pos[2] = 0.5
        gripper_length = 0.08
        move_gripper(env, gripper_length)
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
        current_pos = np.array(obs['ee_pos'])

        target_pos[2] = 0.23
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
        current_pos = np.array(obs['ee_pos'])

        gripper_length = 0.025
        move_gripper(env, gripper_length)

        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.025
        obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pitch=True, target_pitch=np.pi/2)

    elif pred_class == 1:
        target_pos = np.array(des_pos)
        target_pos[2] = 0.5
        gripper_length = 0.025
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
    
    elif pred_class == 2:
        target_pos = np.array(des_pos)
        gripper_length = 0.025
        obs, reward, done, info = move_position(env, target_pos, gripper_length)
        current_pos = np.array(obs['ee_pos'])

        gripper_length = 0.05
        move_gripper(env, gripper_length)

        target_pos = current_pos
        target_pos[2] = 0.5
        gripper_length = 0.08
        obs, reward, done, info = move_position(env, target_pos, gripper_length)

        target_pos = np.array(env.read_debug_parameter())[:3]
        obs, reward, done, info = move_position(env, target_pos, gripper_length)

    elif pred_class == 3:
        target_pos = current_pos
        gripper_length = current_length
        target_joint = list(current_joint_pose.copy())
        target_yaw = current_orn[2]
        if env.robot.arm_upper_limits[5] - target_joint[5] <= 0.01:
            target_joint[5] = env.robot.arm_lower_limits[5]
            target_yaw -= np.pi
        else:
            target_joint[5] += np.pi/2
            target_yaw += np.pi/2
        obs, reward, done, info = move_joint(env, target_joint, gripper_length)
        # obs, reward, done, info = move_position(env, target_pos, gripper_length, config_pos=False, config_orn=True, target_yaw=target_yaw)

    elif pred_class == 4:
        target_pos = current_pos
        gripper_length = current_length
        target_joint = list(current_joint_pose.copy())
        if env.robot.arm_upper_limits[4]- target_joint[4] <= 0.01:
            target_joint[4] = env.robot.arm_lower_limits[4]
        target_joint[4] += np.pi/2
        
        obs, reward, done, info = move_joint(env, target_joint, gripper_length)
        
    else:
        raise ValueError("Invalid class")
    
    return obs, reward, done, info

def stream_joint_pose(env, brick_id):

    ip_address = '127.0.0.1'
    port = '11001'

    # Create a TCP/IP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = (ip_address, int(port))
    print('connect to server at %s port %s' % server_address)
    s.connect(server_address)

    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 100)

    while True:
        joint_pose = env.robot.get_joint_pose()
        gripper_length = env.robot.get_gripper_length()
        joint_pose_degree = np.array(joint_pose)/np.pi*180
        gripper_length = 1 - gripper_length/0.085
        joint_pose_msg = ','.join(list(joint_pose_degree.astype('str'))+[str(gripper_length)])
        msg = f'update_pose {joint_pose_msg}'
        # print(msg)
        return_data = send_message(s, msg)
    # return return_data, msg

def log_robot_object(env, brick_id):
    with open('logs/robot_object.csv', 'w') as f:
        while True:
            joint_pose = np.array(env.robot.get_joint_pose())
            gripper_length = env.robot.get_gripper_length()
            joint_pose = np.array(joint_pose)/np.pi*180
            gripper_length = 1 - gripper_length/0.085
            cubePos, cubeOrn = p.getBasePositionAndOrientation(brick_id)
            cubePos = np.array(cubePos)
            cubeOrn = np.array(cubeOrn)
            joint_pose_msg = ','.join(list(joint_pose.astype('str'))+
                                    [str(gripper_length)]+
                                    list(cubePos.astype('str'))+
                                    list(cubeOrn.astype('str')))
            f.write(joint_pose_msg+'\n')
            time.sleep(0.01)


def send_message(s, msg, ip_address = '127.0.0.1', port = '11001'):  

    # Example message: 
    # update_pose -180,30,75,-10,90,0,1

    # Enum of message types, must match InterboClient
    class MessageType:
        Invalid, Acknowledge, Goodbye, PoseUpdate = range(4)

    # Key Parameters
    default_buffer_size = 1024
    buffer_size = default_buffer_size

    if (msg == "exit"):
        msg = "2"
        s.send(msg.encode('utf-8'))      
    elif ("update_pose" in msg):            
        msgSplit = msg.split(' ')
        if (len(msgSplit) == 2):
            poseArr = msgSplit[1]
            msg = "%s\t%s\n" % (str(int(MessageType.PoseUpdate)), poseArr)
        else:
            msg = ''
            print('Invalid test config')

    s.send(msg.encode('utf-8'))
    
    data = s.recv(buffer_size)
    # print('received: ', data)
    return data

def preprocess(motion_list, max_len):
    tmp_x = np.array(motion_list)[:,:3] # exclude the last dimension corresponding to button pressing
    tmp_x = np.concatenate([tmp_x, np.zeros((max_len - tmp_x.shape[0], tmp_x.shape[1]))]) # zero padding
    # tmp_x_lag = np.concatenate([np.zeros((1, tmp_x.shape[1])), tmp_x[:-1, :]]) # lag 1
    # velocity = tmp_x - tmp_x_lag
    # tmp_x = np.concatenate([tmp_x, velocity], axis=1)
    tmp_x = np.expand_dims(tmp_x, axis=0)
    return tmp_x

def load_model(model_path="best_cnn_model.h5"):
    model = tf.keras.models.load_model(model_path)
    return model

def getting_phantom_pos(UDPServerSocket, bufferSize = 1024):
    global pos_orient
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        # clientMsg = "Message from Client:{}".format(message)
        # clientIP = "Client IP Address:{}".format(address)

        dict_msg = json.loads(message)
        pos_orient = dict_msg["pos_orin"]


def inference(pred_threshold=0.6):
    model = load_model()
    # class_dict = {0:"Class 0: Grasping", 1:"Class 1: Lifting", 2:"Class 2: Moving", 3:"Class 3: Placing"}
    max_len = 110
    x = threading.Thread(target=getting_phantom_pos, args=(1,))
    x.start()

    y = threading.Thread(target=stream_joint_pose, args=(env, s))
    y.start()

    ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'),)
    camera = None
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, ycb_models, camera, vis=True)

    env.reset()
    slave_pos = env.read_debug_parameter()
    obs, reward, done, info = env.step(slave_pos, 'end')
    return_data, msg = stream_joint_pose(env, slave_pos[6])
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
            print(f"Predicted Class = {pred_class}")
            if pred.max() < pred_threshold:
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

def rescale_pos_orient(pos_orient):
    x_range_device = [-0.21, 0.2] # mid val = -0.16 range = 0.14
    y_range_device = [-0.15, 0.15] # mid val = -0.07 range = 0.38
    z_range_device = [-0.11, 0.144] # mid val = 0.31 range = 0.31

    x_range = [-0.224, 0.224]
    y_range = [-0.224, 0.224]
    z_range = [0.0, 1.0]

    x = ((pos_orient[1] - x_range_device[0])/(x_range_device[1] - x_range_device[0]) - 0.5)*0.85
    y = ((pos_orient[0] - y_range_device[0])/(y_range_device[1] - y_range_device[0]) - 0.5)*0.85
    z = ((pos_orient[2] - z_range_device[0])/(z_range_device[1] - z_range_device[0]))*0.7
    # z = pos_orient[2] - 0.2

    return x, y, z