import os

import numpy as np
import pybullet as p
import threading

from tqdm import tqdm
from task_environment import init_env, stream_joint_pose
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math


def user_control_demo():
    # ycb_models = YCBModels(
    #     os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    # )
    # # camera = Camera((1, 1, 1),
    # #                 (0, 0, 0),
    # #                 (0, 0, 1),
    # #                 0.1, 5, (320, 320), 40)
    # camera = None
    # # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    # robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    env = init_env()
    env.reset()
    env.SIMULATION_STEP_DELAY = 0.
    # y = threading.Thread(target=stream_joint_pose, args=(env, ))
    # y.start()
    # env.SIMULATION_STEP_DELAY = 0
    # print(robot.arm_lower_limits)
    # print(robot.arm_upper_limits)
    # joint_pose = [float(pose)*np.pi/180.0 for pose in input("Enter Joint Pose : ").split(',')]
    # for joint in robot.joints:
    #     print(joint.id, joint.name)
    # location = [float(pose) for pose in input("Enter Location : ").split(',')]
    # brick_id = p.loadURDF("meshes/brick/brick_clone.urdf", location, useFixedBase=False)
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
        events = p.getVREvents()
        print(events)
        
          
        # print(obs, reward, done, info)
        # print(obs)
        # print(robot.joints)
        
        # print(robot.arm_controllable_joints)
        # all_joint_pose = robot.get_joint_obs()['positions']
        # joint_pose = [float(pose)*np.pi/180.0 for pose in input("Enter Joint Pose : ").split(',')]
        # print(joint_pose)
        # current_joint_pose = np.array(robot.get_joint_pose())
        # d = np.linalg.norm(np.array(joint_pose)-current_joint_pose)
        # print((np.array(robot.get_joint_pose())/np.pi*180))
        # time.sleep(0.2)
        # robot.move_ee(joint_pose, 'joint')
            # current_joint_pose = np.array(robot.get_joint_pose())
            # d = np.linalg.norm(joint_pose-current_joint_pose)
            
        # print((current_joint_pose*180.0/np.pi).round(2))



if __name__ == '__main__':
    user_control_demo()
