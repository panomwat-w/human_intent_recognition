import json
import socket
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

max_len = 170

localIP = "169.254.74.213"
localIP = "10.248.155.174"
localPort = 20001
msgFromServer = ["Hello UDP Client"]

def load_model(model_path="model/new/best_lstm_model.h5"):
    print('model loading')
    model = tf.keras.models.load_model(model_path)
    x_dummy = np.zeros((40, 3))
    x_dummy = preprocess(x_dummy, max_len)
    model.predict(x_dummy)
    return model

def preprocess(motion_list, max_len):
    tmp_x = np.array(motion_list)[:,:3] # exclude the last dimension corresponding to button pressing
    tmp_x = np.concatenate([tmp_x, np.zeros((max_len - tmp_x.shape[0], tmp_x.shape[1]))]) # zero padding
    tmp_x = np.expand_dims(tmp_x, axis=0)
    return tmp_x

def inference(model, motion_list):
    if len(motion_list) == 0:
        print("No motion data, please try again")
        return -1
    if len(motion_list) > max_len:
        print("Too long, motion data is truncated")
        motion_list = motion_list[:max_len]
    print('Preprocessing')
    x_star = preprocess(motion_list, max_len)
    print('Predicting')
    pred = model.predict(x_star)

    return pred[0]

def send_message(UDPServerSocket, model, bufferSize = 10000):
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        dict_msg = json.loads(message)

        motion_list = np.array(dict_msg["motion"])
        vis = dict_msg["vis"]

        pred = inference(model, motion_list).tolist()
        result = json.dumps({'pred': pred})
        bytesToSend = str.encode(result)
        UDPServerSocket.sendto(bytesToSend, address)
        if vis:
            visualise(motion_list)

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

if __name__ == '__main__':
    # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Bind to address and ip
    UDPServerSocket.bind((localIP, localPort))
    print("UDP server up and listening")
    # model = load_model(model_path="best_cnn_model.h5")
    model = load_model(model_path="best_cnn_model.h5")
    send_message(UDPServerSocket, model, bufferSize = 10000)