import cv2
cap = cv2.VideoCapture('location_cali_cam1.mp4')
fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_count = 0
for i in range(fps):
    ret, frame = cap.read()
    # ret = True
    if ret:
        writeStatus = cv2.imwrite('C:/Users/kim/PycharmProjects/coor/deepsort_v5/gen/' + str(fps_count) + '.jpg', frame)
        print(writeStatus)
        fps_count += 1