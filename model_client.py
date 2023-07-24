import socket
import json
import numpy as np
 
data_dict =  {'motion': (np.random.rand(20, 3)).tolist() }
msgFromClient = json.dumps(data_dict)
bytesToSend         = str.encode(msgFromClient)

serverAddressPort   = ("127.0.0.1", 20001)

bufferSize          = 10000

# Create a UDP socket at client side

UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

 

# Send to server using created UDP socket

UDPClientSocket.sendto(bytesToSend, serverAddressPort)

 

msgFromServer = UDPClientSocket.recvfrom(bufferSize)

 
pred = json.loads(msgFromServer[0])
msg = "Message from Server {}".format(msgFromServer[0])
print(msg)
print(pred)