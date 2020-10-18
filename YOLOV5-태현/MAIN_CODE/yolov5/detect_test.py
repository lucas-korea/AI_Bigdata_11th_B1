import argparse

from utils.datasets import *
from utils.utils import *

def detect():
    weights = 'yolov5s.pt'
    webcam = 0
    img_size = 640
    conf_thres = 0.4
    iou_thres = 0.7
    fourcc = 'mp4v'
    view_img = 'store_ture'

    model = torch.load(weights,map_location=device)['model'].float()

    view_img = True
    torch.backends.cudnn.benchmark = True
    dataset = LoadStreams('0', img_size= img_size)

