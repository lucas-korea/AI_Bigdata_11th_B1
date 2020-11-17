import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import webbrowser
import PIL.Image as pl

#FaceDetection XML 파일
face_classifier =cv2.CascadeClassifier('haarcascade_frontalfac'
                                       'e_default.xml')
#img를 넣으면 face만 잘라 return
def face_extractor(img):
    #사진 흑백 처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #FaceDetection
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    #찾은 얼굴이 없으면 None Return
    if faces is():
        return None

    #찾은 얼굴이 있으면 해당 얼굴 크리만큼 cropped_face에 저장장
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


# webcam 활성화
cap = cv2.VideoCapture(0)
count=0
if cap.isOpened():
    print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))



while True:
    ret, frame = cap.read()
    #얼굴을 감지하면 count 증가 및 200,200사이즈로 사진 수정 후 저장
    if ret:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 'q':
            break
    face_extractor(frame)
    print(count)

    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face_extractor(frame), cv2.COLOR_BGR2GRAY)

        print(count)
        file_name_path = 'faces/user' + str(count) + '.jpg'

        cv2.imwrite(file_name_path, face)
        cv2.imshow('Face Cropper', frame)
        if count==5:
            break

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    else:
        print("Face not Found")
        pass



cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
