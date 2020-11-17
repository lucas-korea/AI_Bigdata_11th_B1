import time
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer, Qt, QCoreApplication, QDateTime
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel, QVBoxLayout, QDesktopWidget, QMainWindow, \
    QTextEdit
import threading



xy_list = []
people = '0'

isFaint = 0
isViolence = 0


v = open("C:/Users/kim/Documents/카카오톡 받은 파일/violence_real2.txt", 'r')
f = open("C:/Users/kim/Documents/카카오톡 받은 파일/시연영~2/시연영상용/violence_real2.txt", 'r')
f1 = open('faint.txt','r')
data = f.read()
data_splited = data.split('\n\n')
# print(data_splited)
data_splited_obj = []
data_splited_obj_ap = []


def compare12(l1,l2):
    new_list = []
    mm = []
    for m in l2:
        min = 100000000
        for idx,n in enumerate(l1):
            s = (m[0]-n[0])**2 + (m[1]-n[1])**2
            if s < min:
                min = s
                if idx not in mm:
                    mm.append(idx)
    for m in mm:
        tmp=[l1[m][0],l1[m][1]]
        new_list.append(tmp)
    return new_list


def getMap(x_c,y_c):
    new_x = 0
    new_y = 0
    new_coor = []

    # 90 < y < 600
    if 140 <= y_c and y_c < 660:

        a = (38360 - 55 * y_c) / 84
        b = (5 * y_c + 37100) / 42

        if a <= x_c and x_c <= b:
            new_x = 920 - (755 * (y_c - 140)) / 420
            new_y = 80 + (95 * (x_c - a)) / (b - a)
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

def loadXY():
    global xy_list
    global people
    global isViolence
    global isFaint



    while True:
        # human도 여기서 나누고
        for j in range(len(data_splited) - 1):  # frame
            data_splited_obj = data_splited[j].split('\n')
            # print('=======================')
            # print(data_splited_obj)
            obj_label = []

            for a in range(len(data_splited_obj)):  # of obj

                # print(a)
                # print((data_splited_obj[a]))

                if (data_splited_obj[0] == 'None'):
                    # print('None')
                    obj_label.append('None')
                else:
                    data_splited_obj_dummy = data_splited_obj[a].split(' ')
                    obj_label.append(data_splited_obj_dummy[0])
                    obj_label.append(data_splited_obj_dummy[1])
                    obj_label.append(data_splited_obj_dummy[2])
            data_splited_obj_ap.append(obj_label)

        # 판단 시작
        faint_flag = 0
        faint_frame = 0
        faint_stack  = 0
        violence_flag = 0
        violence_frame = 0
        violence_stack = 0

        num0 = '0'
        num1 = '1'
        num2 = '2'
        for i in range(len(data_splited_obj_ap)):
            del xy_list[:]
            tmp1=[]
            tmp2 = []


            #한 프레임에 다수의 인원들 x,y,isfaint, isviolence 리스트를 xy_list에 append
            for j in range(len(data_splited_obj_ap[i])//3):
                isFaint = 0
                isViolence = 0
                faint_flag=0
                if num1 in data_splited_obj_ap[i][j*3:3*j+3] and violence_flag < 2: # 상황이 일어났는지 보고, 만약 real 상황으로 판단하면 더이상 판단하지 않음
                    if violence_flag == 0:
                         violence_frame = i
                    violence_flag = 1 # unsure violence_상황
                    violence_stack +=1
                    if violence_stack >= 20:
                        violence_flag = 2 # real violence 상황으로 판단
                    if (i - violence_frame) > 100: # 100 frame 내로 상황 판단 못할시 strack 데이터 무시
                        violence_stack = 0
                        violence_flag = 0
                if num2 in data_splited_obj_ap[i][j*3:3*j+3] and faint_flag < 2: # 상황이 일어났는지 보고, 만약 real 상황으로 판단하면 더이상 판단하지 않음
                    if faint_flag == 0:
                         faint_frame = i
                    faint_flag = 1 # unsure violence_상황
                    faint_stack +=1
                    if faint_stack >= 20:
                        faint_flag = 2 # real violence 상황으로 판단
                    if (i - faint_frame) > 100: # 100 frame 내로 상황 판단 못할시 strack 데이터 무시
                        faint_stack = 0
                        faint_flag = 0

                # print('frame: ', i, 'faint flag   :', faint_flag, '  faint stack    :', faint_stack)
                # print('frame: ', i, 'violence_flag:', violence_flag, '  violence_stack :', violence_stack)


                if faint_flag >=2 :
                    isFaint = 1
                if violence_flag >=2 :
                    isViolence = 1

                print(data_splited_obj_ap[i][j*3+1],data_splited_obj_ap[i][j*3+2])
                new_coor = getMap(float(data_splited_obj_ap[i][j*3+1]),float(data_splited_obj_ap[i][j*3+2]))
                if len(new_coor) == 0:
                    continue
                p = [new_coor[0],new_coor[1],isFaint,isViolence ]
                xy_list.append(p)


            time.sleep(0.03)

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
        #self.ex = 1

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
        self.label1.setStyleSheet("color: black;"
                                  "border-style: dashed;"
                                  "border-width: 5px;"
                                  "border-color: black;"
                                  "border-radius: 3px")

        self.label4 = QLabel(people, self)
        self.label4.move(590, 450)
        self.font4 = self.label4.font()
        self.font4.setPointSize(40)
        self.label4.setFont(self.font4)
        self.label4.setStyleSheet("color: black;"
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
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout_run)


    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw_rect(qp)
        qp.end()

    #********* 좌표리스트 반복을 통해 지도에 표시하기 #*********
    def draw_rect(self, qp):
        global people
        # qp.setPen(QPen(Qt.blue, 30))
        # qp.drawPoint(990, 820)
        # if 돌발상황 체크
        # else면 좌표찍기
        # ******** 비상사태 지도에 표시 ************#
        # if self.ex == 1:
        #     qp.setPen(QPen(Qt.red, 5))
        #     qp.drawRoundedRect(500, 550, self.emer, self.emer,50,50)
        # ************************************************#

        # qp.setPen(QPen(Qt.blue, 8))
        # qp.drawPoint(self.pos_x, self.pos_y)
        #
        # qp.setPen(QPen(Qt.red, 8))
        # qp.drawPoint(self.pos_x+10, self.pos_y+10)

        # ******** 좌표리스트 돌면서 id랑 해당좌표를 지도에 표시 ************#
        global title
        #title+='d'
        # for m in self.map:
        #     # 점으로 사람 표시
        #     qp.setPen(QPen(Qt.blue, 10))
        #     qp.drawPoint(m[0], m[1])
        #
        #     # id표시
        #     qp.setFont(QFont('Consolas', 14))
        #     qp.drawText(m[0], m[1], title) # (x, y, id)
        # ************************************************#

        # 돌면서 기절,폭력 상황일 때는 표시!
        #print(xy_list)
        for m in xy_list:
            if m[2]==1 or m[3]==1:
                qp.setPen(QPen(Qt.red, 20))
                qp.drawPoint(float(m[0]), float(m[1]))
            else:
                qp.setPen(QPen(Qt.blue, 20))
                qp.drawPoint(float(m[0]),float(m[1]))
        people = str(len(xy_list))


    # ******** 바운딩박스 좌표 불러오는 함수 ************#
    def timeout_run(self):
        global title
        global people
        global xy_list
        global f2
        global f1
        global isFaint
        global isViolence
        # 일단 class 밖 리스트변수에 좌표반환값 저장하고
        # timeout_run


        # 돌발상황이 기존좌표와 똑같다면 크기조절
        # if self.pos_x < 0:
        #     self.x_direction *= -1
        #
        # if self.pos_x == 700:
        #     self.y_direction = -1
        #     self.x_direction = 0
        #

        #title += 'd'

        # ******** 비상상황 표시 움직이게 하는 코드************#
        if self.emer == 50:
            self.emer_dir = -1
        if self.emer == 40:
            self.emer_dir = 1
        #************************************************#

        self.emer = self.emer+(self.emer_dir*self.emer_speed)
        # self.pos_x = self.pos_x + (self.x_direction * self.speed)
        # self.pos_y = self.pos_y + (self.y_direction * self.speed)
        # print(self.pos_x, self.pos_y)

        # map list에 서버에서 좌표를 받아와 추가하는 코드
        #id_loc = []
        #self.map.append(id_loc)

        #
        # # 임의 리스트의 위치를 변환시키기 위한 코드이고 필요없음
        # for m in self.map:
        #     m[0] = m[0]+0.5


        #***** 비상상황이라면 숨겨져 있던 label1에 text와 color를 바꿈 #*****
        if isFaint ==1 :
            self.label1.setText('     Faint')
            self.label1.setStyleSheet('color: red;'
                                      'background-color:#FA8072')

        if isViolence ==1 :
            self.label1.setText('  Violence')
            self.label1.setStyleSheet('color: red;'
                                      'background-color:#FA8072')

        # ************************************************#
        if int(people) != 0:
            self.label4.setText(people)
            self.label4.setStyleSheet("color: blue;"
                                      "border-style: dashed;"
                                      "border-width: 5px;"
                                      "border-color: black;"
                                      "border-radius: 3px")

        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())