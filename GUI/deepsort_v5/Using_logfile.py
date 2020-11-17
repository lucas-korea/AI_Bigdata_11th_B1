import time
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer, Qt, QCoreApplication, QDateTime
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel, QVBoxLayout, QDesktopWidget, QMainWindow, \
    QTextEdit
import threading


f1 = open('walking_cross_people_cam1.txt','r')
f2 = open('walking_cross_people_cam2.txt','r')


xy_list = []
people = '0'

def getMap_cam1(x_c,y_c):
    new_x = 0
    new_y = 0
    new_coor=[]

    # 90 < y < 600
    if 140 <= y_c and y_c < 660:

        a = (38360-55*y_c) / 84
        b = (5*y_c + 37100) / 42

        if a <= x_c and x_c <= b:
            new_x = 920 - (755*(y_c-140)) / 420
            new_y = 80 + (95*(x_c-a)) / (b-a)
            new_coor = [new_x, new_y]
            return new_coor

        else:
            return new_coor

    elif 50 <= y_c and y_c < 140:
        a = (85455 - 394*y_c) / 83
        b = (81700 - 50*y_c) / 83
        c = ((26*y_c + 13000) /19) - 835

        if  a <= x_c and x_c <= b:
            new_x = 860 - (c*(y_c-57))/83
            new_y = 80 + 95*(x_c-a) / (b-a)
            new_coor = [new_x, new_y]
            return new_coor

        else:
            return new_coor

    else:
        return new_coor




def getMap_cam2(x_c,y_c):
    new_x = 0
    new_y = 0
    new_coor=[]

    # 90 < y < 600
    if 210 <= y_c and y_c < 930:

        a = (59430-43*y_c) / 72
        b = (31*y_c + 41010) / 36

        if a <= x_c and x_c <= b:
            new_x = 835 + (140/(b-a)) * (x_c - a)
            new_y = (610/720)*(y_c-219) + 210
            new_coor = [new_x, new_y]
            return new_coor

        else:
            return new_coor
    elif 148<= y_c and y_c < 210:
        a = ((10*y_c)+19600) /31
        b = (215*y_c - 4230) / 31

        if  a <= x_c and x_c <= b:
            new_x = 825 + (((y_c + 780)-825)/ (b-a)) * (x_c -a)
            new_y = (130/62) * (y_c-148) + 80
            new_coor = [new_x, new_y]
            return new_coor

        else:
            return new_coor

    else:
        return new_coor





def loadXY():
    global xy_list
    global people

    while True:
        del xy_list[:]
        frame = []

        for f in f1:
            if f == '\n' or f=='None\n':
                break

            id, x, y = f.split()

            if 835<int(x) and int(x)<990:
                break

            frame.append([int(x), int(y),1])


        for f in f2:
            if f == '\n' or f=='None\n':
                break
            id, x, y = f.split()
            frame.append([int(x), int(y),2])

        # new_x, new_y = getMap(x,y)
        # frame에 새로 넣기
        if len(frame)==0:
            continue
        for f in frame:
            if f[2] == 1:
                coor = getMap_cam1(f[0],f[1])
                if f[0]>=800 and f[1]<=170:
                    continue
                # if f[0]>=950 or f[0]<=90:
                #     continue
                if f[1]>=560 or f[1]<100:
                    continue
                if f[0]>365 and f[1]<140:
                    continue
                if f[1] < 200:
                    continue
                print('1')
                print(f[0],f[1])
                xy_list.append([coor[0],coor[1]])

            elif f[2]==2:
                if f[0] < 270 or f[0] >1900:
                    continue
                if f[1] < 130 and f[0] >1000:
                    continue
                if f[1]<148:
                    continue
                if f[0]<700 and f[1]<210:
                    continue
                if f[1]>930:
                    continue
                if f[1] <=156:
                    continue
                coor = getMap_cam2(f[0],f[1])
                print('2')
                print(f[0], f[1])
                xy_list.append([coor[0],coor[1]])


        time.sleep(0.1)

t1 = threading.Thread(target=loadXY)
t1.start()






class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 서버에서 받아온 좌표를 저장하는 리스트
        self.map = [[500,500,'Bomin'],[550,550,'Dabin']]

        self.pos_x = int(self.width() / 2)
        self.pos_y = 550
        self.x_direction = 1
        self.y_direction = 0
        self.emer = 50
        self.emer_dir = 1
        self.speed = 1
        self.emer_speed = 0.5

        # check 'is비상상황'
        self.ex = 0

    def initUI(self):
        self.setGeometry(300, 100, 1100, 900)

        # 배경화면 설정
        oImage = QImage("C:/Users/kim/Desktop/background.png")
        sImage = oImage.scaled(1100, 900)
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)

        # 비상상황 알림 표시 label
        self.label1 = QLabel('  N O N E  ', self)
        self.label1.move(70, 450)
        self.font1 = self.label1.font()
        self.font1.setPointSize(40)
        self.label1.setFont(self.font1)
        self.label1.setStyleSheet( "color: black;"
                              "border-style: dashed;"
                              "border-width: 5px;"
                              "border-color: black;"
                              "border-radius: 3px")


        self.label4 = QLabel(people, self)
        self.label4.move(590, 450)
        self.font4 = self.label4.font()
        self.font4.setPointSize(40)
        self.label4.setFont(self.font4)
        self.label4.setStyleSheet( "color: black;"
                              "border-style: dashed;"
                              "border-width: 5px;"
                              "border-color: black;"
                              "border-radius: 3px")

        self.label2 = QLabel(' 상 황 ', self)
        self.label2.move(180, 350)
        self.font2 = self.label2.font()
        self.font2.setPointSize(35)
        self.label2.setFont(self.font2)
        self.label2.setStyleSheet("color: red;"
                                  )

        self.label3 = QLabel(' 인 원', self)
        self.label3.move(550, 350)
        self.font3 = self.label3.font()
        self.font3.setPointSize(35)
        self.label3.setFont(self.font3)
        self.label3.setStyleSheet("color: red;"
                                  )

        self.setWindowTitle('drawRect')
        self.timer = QTimer(self)
        self.timer.start(100)
        self.timer.timeout.connect(self.timeout_run)


    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw_rect(qp)
        qp.end()

    #********* 좌표리스트 반복을 통해 지도에 표시하기 #*********
    def draw_rect(self, qp):
        global people

        # ******** 비상사태 지도에 표시 ************#
        if self.ex == 1:
            qp.setPen(QPen(Qt.red, 5))
            qp.drawRoundedRect(500, 550, self.emer, self.emer,50,50)
        # ************************************************#


        for m in xy_list:
            qp.setPen(QPen(Qt.blue, 20))
            qp.drawPoint(int(m[0]), int(m[1]))

        people = str(len(xy_list))

    # ******** 바운딩박스 좌표 불러오는 함수 ************#
    def timeout_run(self):
        global title
        global people
        global xy_list
        global f2
        global f1
        # 일단 class 밖 리스트변수에 좌표반환값 저장하고
        # timeout_run


        # ******** 비상상황 표시 움직이게 하는 코드************#
        if self.emer == 50:
            self.emer_dir = -1
        if self.emer == 40:
            self.emer_dir = 1
        #************************************************#



        # 임의 리스트의 위치를 변환시키기 위한 코드이고 필요없음
        for m in self.map:
            m[0] = m[0]+0.5


        #***** 비상상황이라면 숨겨져 있던 label1에 text와 color를 바꿈 #*****
        if self.ex ==1 :
            self.label1.setText('Emergency')
            self.label1.setStyleSheet("color: red;"
                                      "border-style: dashed;"
                                      "border-width: 5px;"
                                      "border-color: red;"
                                      "border-radius: 3px")

        # ************************************************#
        self.label4.setText(people)

        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())




















