import numpy as np
import os

np.random.seed(3)
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 원본 이미지 위치
optInputPath = './sample'
# 늘릴 이미지가 저장될 위치
optOutputPath = './preview/'

# 이미지 크기 조정 비율
optRescale = 1. / 255
# 이미지 회전
optRotationRange = 15
# 이미지 수평 이동
optWidthShiftRange = 0.1
# 이미지 수직 이동
optHeightShiftRange = 0.1
# 이미지 밀림 강도
optShearRange = 0.5
# 이미지 확대/ 축소
optZoomRange = [0.8, 2.0]
# 이미지 수평 뒤집기
optHorizontalFlip = True
# 이미지 수직 뒤집기
optVerticalFlip = True
optFillMode = 'nearest'
# 이미지당 늘리는 갯수
optNbrOfIncreasePerPic = 3
# 배치 수
optNbrOfBatchPerPic = 70

'''
 총 개수 optNbrOfIncreasePerPic * optNbrOfBatchPerPic 
 예 >
 사진 1장에 
 optNbrOfIncreasePerPic = 5
 optNbrOfBatchPerPic = 5
 = 1 * 3 * 70 = 210장 생성  
'''

# 데이터셋 불러오기
train_datagen = ImageDataGenerator(rescale=optRescale,
                                   rotation_range=optRotationRange,
                                   width_shift_range=optWidthShiftRange,
                                   height_shift_range=optHeightShiftRange,
                                   shear_range=optShearRange,
                                   zoom_range=optZoomRange,
                                   horizontal_flip=optHorizontalFlip,
                                   vertical_flip=optVerticalFlip,
                                   fill_mode=optFillMode)
'''
폴더가 존재하는지 확인하고
없다면 생성 
'''


def checkFoler(path):
    try:
        if not (os.path.isdir(path)):
            os.makedirs(os.path.join(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def increaseImage(path, folder):
    for index in range(0, optNbrOfIncreasePerPic):
        img = load_img(path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        # 부풀리는 이미지를 저장할 폴다가 존재하는지 확인
        # 없다면 생성
        checkFoler(optOutputPath + folder)
        print('index : ' + str(index))
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=optOutputPath + folder, save_prefix='products',
                                        save_format='jpg'):
            i += 1
            print(folder + " " + str(i))
            if i >= optNbrOfBatchPerPic:
                break


def generator(dirName):
    checkFoler(optOutputPath)
    try:
        fileNames = os.listdir(dirName)
        for fileName in fileNames:
            fullFileName = os.path.join(dirName, fileName)
            if os.path.isdir(fullFileName):
                generator(fullFileName)
            else:
                # 확장자
                ext = os.path.splitext(fullFileName)[-1]
                # 폴더 이름
                folderName = os.path.splitext(fullFileName)[0].split('/')[-2]
                if (ext == '.jpg'):
                    increaseImage(fullFileName, folderName)

    except PermissionError:
        pass


if __name__ == "__main__":
    generator(optInputPath)