from os import listdir
from os.path import isdir
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN


model = load_model('facenet_keras.h5')
print(model.inputs)
print(model.outputs)
#모델은 160x160 형태의 정사각형 이미지. 128개의 벡터로 이루어진 얼굴 출력

#사진에서 하나의 얼굴 추출(MTCNN사용)
def extract_face(filename, required_size=(160, 160)):
    # 파일로 이미지 불러오기
    image = Image.open(filename)
    # RGB로 변환, 필요시
    image = image.convert('RGB')
    #이미지를 배열로 변환
    pixels = np.asarray(image)
    #pixels = pyplot.imread(filename) #전기수에서는 에러이 변환 안하고 이거 사용. 확인필요


    # 디텍터 생성,MTCNN으로 얼굴 감지 클래스 만들기
    detector = MTCNN()
    # 이미지에서 얼굴 감지
    results = detector.detect_faces(pixels)

    #얼굴 감지 실패 시 0으로 채우기
    if results == []:
        return np.zeros((160, 160, 3))

    # 첫번째 얼굴에서 경계 상자 추출
    x1, y1, width, height = results[0]['box']
    # 음의 픽셀 방지하려고 절댓값 사용
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # 얼굴 추출
    face = pixels[y1:y2, x1:x2]

    # 얼굴 이미지 크기를 160x160으로 resize
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    print("image before np", image)
    face_array = np.asarray(image)
    print("extract face produces", face_array.shape)
    return face_array

# 메인에서 이거로 테스트해보기 folder = '5-celebrity-faces-dataset/train/ben_afflek/'

# def plot_images(folder, plot_h, plot_w):
#     # folder = '5-celebrity-faces-dataset/train/ben_afflek/'
#     i = 1
#     # 파일 쭉 나열
#     for filename in listdir(folder):
#         path = folder + filename
#         # path엥서 얼굴 추출
#         face = extract_face(path)
#         print(i, face.shape)
#         # 플랏하기 2x7으로
#         pyplot.subplot(plot_h, plot_w, i)
#         pyplot.axis('off')
#         pyplot.imshow(face)
#         i += 1
#     pyplot.show()
#     #이거의 결과가 2x7로 얼굴만 뽑아낸 사진 나올거임.



#directory 예시= 5-celebrity-faces-dataset/train/ben_afflek/
# 디렉토리 안의 모든 이미지를 불러오고 이미지에서 얼굴 추출

def load_faces(directory):
    # directory = '5-celebrity-faces-dataset/train/ben_afflek/'
    faces = list()
    # 파일 열거
    for filename in listdir(directory):
        # 경로
        path = directory + filename
        # 얼굴 추출
        face = extract_face(path)
        # 저장
        faces.append(face)
    return faces


#directory 예시= 5-celebrity-faces-dataset/train/
#각 하위 디렉토리에서 얼굴을 감짛고 각 감지된얼굴에 레이블 할당
#이미지를 포함하는 각 클래스에 대해 하나의 하위 디렉토리가 포함된 데이터셋을 불러오기

# direc= '5-celebrity-faces-dataset/train/'
def load_dataset(direc):
    x, y = list(), list()
    #for every class directory in this train/test directory
    for subdir in listdir(direc):

        path = direc + subdir + '/'
        #if it is a file and not a dir then skip
        if not isdir(path):
            continue
        #하위 디렉토리의 모든 얼굴 불러오기
        faces = load_faces(path)
        #라벨링하기
        labels = [subdir for i in range(len(faces))]
        #summarize progress
        print('loaded %d 개 examples for class: %s에서' %(len(faces), subdir))
        print(faces)
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x), np.asarray(y)



#데이터 만들어서 dir 바꾸기
# 훈련 데이터셋 불러오기
trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
print(trainX.shape, trainy.shape)
# 테스트 데이터셋 불러오기
testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
print(testX.shape, testy.shape)
# 배열을 단일 압축 포맷 파일로 저장
np.savez_compressed('B1_Dataset.npz', trainX, trainy, testX, testy)

#이거의 결과로는 train 데이터 셋의 모든 사진을 불러와 93개의 샘플이 생성됩니다. 그런 다음 val 데이터셋을 불러와 테스트 데이터셋으로 사용할 수 있는 25개의 샘플이 제공됩니다.
#이 두 데이터셋은 5-celebrity-faces-dataset.npz라는 걸로 저장.


# 출력ㅇㅖ시
# >14개의 예제를 불러왔습니다. 클래스명: ben_afflek
# >19개의 예제를 불러왔습니다. 클래스명: madonna
# >17개의 예제를 불러왔습니다. 클래스명: elton_john
# >22개의 예제를 불러왔습니다. 클래스명: mindy_kaling
# >21개의 예제를 불러왔습니다. 클래스명: jerry_seinfeld
# (93, 160, 160, 3) (93,)
# >5개의 예제를 불러왔습니다. 클래스명: ben_afflek
# >5개의 예제를 불러왔습니다. 클래스명: madonna
# >5개의 예제를 불러왔습니다. 클래스명: elton_john
# >5개의 예제를 불러왔습니다. 클래스명: mindy_kaling
# >5개의 예제를 불러왔습니다. 클래스명: jerry_seinfeld
# (25, 160, 160, 3) (25,)

#이제 이것들은 얼굴 감지 모델로 제공됨