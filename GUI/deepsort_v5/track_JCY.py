from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
# https://github.com/pytorch/pytorch/issues/3678
import sys
import os



import sys
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer, Qt, QCoreApplication, QDateTime
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel, QVBoxLayout, QDesktopWidget, QMainWindow, \
    QTextEdit
import threading



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.insert(0, './yolov5')


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


XY = []
people = '0'

def getMap(x_c,y_c):
    new_x = 0
    new_y = 0
    new_coor=[]

    # 90 < y < 600
    if 140 <= y_c and y_c < 560:

        a = (38360-55*y_c) / 84
        b = (5*y_c + 37100) / 42

        if a <= x_c and x_c <= b:
            new_x = 875 - (755*(y_c-140)) / 420
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


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt, save_img=False):

    global XY
    global people

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                del XY[:]
                people = str(len(det))
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    # bounding box x,y
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    print('x_c : %s, y_c : %s, bbox_w : %s, bbox_h : %s' % (x_c, y_c, bbox_w, bbox_h))
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    # x,y 좌표를 parameter로 계산해서 맵에 표시할 새로운 좌표 반환
                    map_coor = getMap(x_c, y_c)
                    if len(map_coor) == 0:
                        continue
                    print('x_new : %s, y_new : %s' % (map_coor[0], map_coor[1]))
                    new_XY = map_coor
                    # global map x,y list
                    XY.append(new_XY)

                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                print('')
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:  
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))













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
        self.label1 = QLabel('  N O N E   ', self)
        self.label1.move(90, 450)
        self.font1 = self.label1.font()
        self.font1.setPointSize(40)
        self.label1.setFont(self.font1)

        self.label4 = QLabel(people, self)
        self.label4.move(660, 450)
        self.font4 = self.label4.font()
        self.font4.setPointSize(40)
        self.label4.setFont(self.font4)

        self.label2 = QLabel(' 비 상 상 황', self)
        self.label2.move(100, 350)
        self.font2 = self.label2.font()
        self.font2.setPointSize(35)
        self.label2.setFont(self.font2)

        self.label3 = QLabel(' 인 원', self)
        self.label3.move(600, 350)
        self.font3 = self.label3.font()
        self.font3.setPointSize(35)
        self.label3.setFont(self.font3)

        self.setWindowTitle('drawRect')
        self.timer = QTimer(self)
        self.timer.start(1000/30)
        self.timer.timeout.connect(self.timeout_run)


    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.draw_rect(qp)
        qp.end()

    #********* 좌표리스트 반복을 통해 지도에 표시하기 #*********
    def draw_rect(self, qp):


        for m in XY:
            qp.setPen(QPen(Qt.blue, 30))
            qp.drawPoint(m[0], m[1])


    # ******** 바운딩박스 좌표 불러오는 함수 ************#
    def timeout_run(self):
        global title
        global people
        # 일단 class 밖 리스트변수에 좌표반환값 저장하고
        # timeout_run




        # 임의 리스트의 위치를 변환시키기 위한 코드이고 필요없음
        for m in self.map:
            m[0] = m[0]+0.5


        #***** 비상상황이라면 숨겨져 있던 label1에 text와 color를 바꿈 #*****
        if self.ex ==1 :
            self.label1.setText('Emergency')
            self.label1.setStyleSheet('color: red;'
                                      'background-color:#FA8072')

        # ************************************************#
        self.label4.setText(people)

        self.update()



def loadCoor():
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())

t = threading.Thread(target=loadCoor)
t.start()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='gen/2.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)



