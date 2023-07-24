import json
import socket
import numpy as np
import tensorflow as tf

max_len = 170

localIP = "127.0.0.1"
localPort = 20001
msgFromServer = ["Hello UDP Client"]

def load_model(model_path="best_lstm_model.h5"):
    print('model loading')
    model = tf.keras.models.load_model(model_path)
    x_dummy = np.zeros((20, 40, 3))
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

        pred = inference(model, motion_list).tolist()
        result = json.dumps({'pred': pred})
        bytesToSend = str.encode(result)
        UDPServerSocket.sendto(bytesToSend, address)

if __name__ == '__main__':
    # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Bind to address and ip
    UDPServerSocket.bind((localIP, localPort))
    print("UDP server up and listening")
    model = load_model(model_path="best_lstm_model.h5")
    send_message(UDPServerSocket, model, bufferSize = 10000)