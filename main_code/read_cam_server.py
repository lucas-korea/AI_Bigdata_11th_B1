import cv2
import socket
import numpy
from queue import Queue
from _thread import *


def threaded(client_socket, addr, queue):
    print('Connected by :', addr[0], ':', addr[1])
    while True:
        try:
            data = client_socket.recv(1024)

            if not data:
                print('Disconnected by ' , + addr[0], ':', addr[1])
                break
            stringData = queue.get()
            client_socket.send(str(len(stringData)).ljust(16).encode())
            client_socket.send(stringData)
        except ConnectionAbortedError as e:

            print('Disconnected by '+ addr[0], ':', addr[1])
            break
    client_socket.close()

def read_cam(queue, num = 0):

    capture = cv2.VideoCapture(num)

    while True:
        ret, frame = capture.read()

        if ret == False:
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)

        data = numpy.array(imgencode)
        stringData = data.tostring()

        queue.put(stringData)

        cv2.imshow('image', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

if __name__ == '__main__':
    enclosure_queue = Queue()
    HOST = '127.0.0.1'
    PORT = 9999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST,PORT))
    server_socket.listen()

    print('server start')
    start_new_thread(read_cam, (enclosure_queue,))

    while  True:
        print('wait')
        client_socket, addr = server_socket.accept()
        start_new_thread(threaded, (client_socket, addr, enclosure_queue,))

    server_socket.close()

