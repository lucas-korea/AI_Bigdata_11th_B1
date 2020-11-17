import argparse

from utils.datasets import *
from utils.utils import *
import multiprocessing
from socket import *
from detect2 import *


# def get_coordinate():
# global xy

def detect(save_img=False,):
    # global xy
    #  xy = []
    # port = 8080
    #
    # clientSock = socket(AF_INET, SOCK_STREAM)
    # clientSock.connect(('141.223.140.8', port))
    #
    # print('접속 완료')
    # recvData = clientSock.recv(1024)
    # print('상대방 :', recvData.decode('utf-8'))

    # out = 'inference/output'
    # source = 'inference/images'
    # weights = weights
    # view_img = 'store_ture'
    # save_txt = 'store_true'


    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') #webcam = sorce 인자를 1로 주게 되면 내장 캠이 아닌 외장 캠 사용가능 단, dataset.py에서 코드수정 필요
    save_txt = True
    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams('0', img_size=imgsz) #캠 온
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time() #학습 시간 측정
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        # test print("path :",path,"img :",img, "im0s : ",im0s, "vid_cap :", vid_cap)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print('pred:', pred)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        flag_detect = 0
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            # print('s : ', s) # 그냥 shape
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                flag_detect = 1
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # print('n : ',n)
                    s += '%g %ss, ' %  (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    # xy.append(xyxy)
                    # print(xy[-1])
                    _xyxy = list(xyxy)
                    lngth = len(_xyxy)
                    for i in range(lngth):
                        sendData = '사람 ' +str(i)+ ' ' + str(xyxy[i])
                        # clientSock.send(sendData.encode('utf-8'))


                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xyxy2 = xyxy.tolist()
                        # xywh = xywh * gn
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 3 + '%.2f\n') % (cls, (xyxy[0].tolist()+xyxy[2].tolist())*0.5, (xyxy[1].tolist()+xyxy[3].tolist())*0.5, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                    file.write('\n')  # label format


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            if flag_detect == 0:
                with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                    file.write('None\n\n')  # label format

            string = s

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(300) == ord('q'):  # q를 누르면 종로하면서 상품 합계 출력.
                    i = 0
                    for target in classification:
                        _cnt = string.count(str(target))
                        total += _cnt * price[i]
                    print("합계는 ", total ,"원 입니다.")
                    time.sleep(3)
                    return 0

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
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

# python detect.py --source ./inference/images/ --weights yolov5s.pt --conf 0.4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='last.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='violence_real3.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output foldepr')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold') #default 0.4
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS') ##default 0.4
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()


