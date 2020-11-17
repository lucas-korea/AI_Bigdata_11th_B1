import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
from sklearn.svm import SVC
from svm import model as sv
from svm import out_encoder
import cv2
import argparse
import sys

#이거 임베딩파일에 있음
def get_embedding(model, face_pixels):
    # 픽셀 값의 척도
    face_pixels = face_pixels.astype('float32')
    # 채널 간 픽셀값 표준화(전역에 걸쳐)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # print(face_pixels.shape)

    # 얼굴을 하나의 샘플로 변환
    # expand dims adds a new dimension to the tensor
    samples = np.expand_dims(face_pixels, axis=0)
    # print(samples.shape)

    # 임베딩을 갖기 위한 예측 생성
    yhat = model.predict(samples)
    return yhat[0]


def main():
    # filename = args["input_file"] #이건 뭐지?
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model = load_model('facenet_keras.h5')

    cap = cv2.VideoCapture(0)
    count = 0
    count_wel=0

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=3)  # face structure

        if len(faces)!=0:
            for (x,y,w,h) in faces:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # take the face pixels from the frame
                crop_frame = frame[y:y + h, x:x + w]  # turn the face pixels back into an image
                new_crop = Image.fromarray(crop_frame)  # resize the image to meet the size requirment of facenet
                new_crop = new_crop.resize((160, 160))  # turn the image back into a tensor
                crop_frame = np.asarray(new_crop)  # get the face embedding using the face net model
                face_embed = get_embedding(model, crop_frame)  # it is a 1d array need to reshape it as a 2d tensor for svm
                face_embed = face_embed.reshape(-1, face_embed.shape[0])  # predict using our SVM model
                pred = sv.predict(face_embed)  # get the prediction probabiltiy
                pred_prob = sv.predict_proba(face_embed)  # pred_prob has probabilities of each class

                # get name
                class_index = pred[0]
                class_probability = pred_prob[0, class_index] * 100
                predict_names = out_encoder.inverse_transform(pred)
                text = '%s' % (predict_names[0])

                if class_probability > 99.5:
                    count_wel+=1
                    cv2.putText(frame, text, (x+30, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Welcome", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    count=0

                elif class_probability<98:
                    count += 1

                if count > 3:
                    cv2.putText(frame, "Please register as a member", (80, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            count_wel=0
            cv2.putText(frame, "Please come closer", (150, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if count_wel > 1:
            cv2.putText(frame, "   ", (250, 80), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 2)

            print(count_wel)

        cv2.imshow('Face Cropper', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
