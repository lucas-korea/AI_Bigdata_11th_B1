#   facenet
#   Modified by Jongha
#   Last Update: 2020.06.02

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer


# get the face embedding for one face
def get_embedding(model, face_pixels):
	# 픽셀 값의 척도
	face_pixels = face_pixels.astype('float32')
	# 채널 간 픽셀값 표준화(전역에 걸쳐)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	print(face_pixels.shape)

    # 얼굴을 하나의 샘플로 변환
    # expand dims adds a new dimension to the tensor
	samples = np.expand_dims(face_pixels, axis=0)
	print(samples.shape)

    # 임베딩을 갖기 위한 예측 생성
	yhat = model.predict(samples)
	return yhat[0]

# 얼굴 데이터셋 불러오기
data = np.load('B1_Dataset.npz')
trainX, trainy, valX, valy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, valX.shape, valy.shape)


model = load_model('facenet_keras.h5')
print('model loaded')


# 훈련 셋에서 각 얼굴을 엠베딩으로 변환하기
newTrainX = list()
for face_pixels in trainX:
	print(face_pixels.shape)
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
#
# 검증 셋에서 각 얼굴을 엠베딩으로 변환하기
newValX = list()
for face_pixels in valX:
    embedding = get_embedding(model, face_pixels)
    newValX.append(embedding)
newValX = np.asarray(newValX)
print(newValX.shape)
#
#
#
# 벡터가 unit이 될 때까지 scale하기. 이거 여기서말고 임베딩할때 추가하자
# in_encoder = Normalizer(norm='l2')
# newTrainX = in_encoder.transform(newTrainX)
# newValX = in_encoder.transform(newValX)

# 배열을 하나의 압축 포맷 파일로 저장
np.savez_compressed('B1_EmbeddingData.npz', newTrainX, trainy, newValX, valy)

# # from now on, we get face embeddings for each face image
# # now we can use these to make predictions using an SVM.
# # Next, open SVMclassifier.
