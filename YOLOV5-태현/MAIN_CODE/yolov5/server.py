from socket import *


port = 8080

serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind(('', port))
serverSock.listen(1)

print('%d번 포트로 접속 대기중...'%port)

connectionSock, addr = serverSock.accept()

print(str(addr), '에서 접속되었습니다.')

while True:
    sendData = input('>>>')
    connectionSock.send(sendData.encode('utf-8'))

    recvData = connectionSock.recv(1024)
    print('상대방 :', recvData.decode('utf-8'))