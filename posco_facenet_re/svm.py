import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice

# 신원 미상의 새로운 이미지가 들어왔을 시 얼굴 분류를 위한 서포트 벡터 머신 동작코드. 새로운 이미지의 얼굴 Embedding 을 추출하여 방문자 얼굴 Embedding 데이터셋 5-celebrity-faces-embeddings.npz  과 비교하여 가장 높은 마진을 가진 얼굴 데이터 라벨을 할당하여 분류
# 5명의 유명인사 얼굴 데이터셋에서 알려진 유명인 중 하나로 얼굴 분류하는 모델 만들기

data = np.load('B1_EmbeddingData.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: training examples =', trainX.shape, 'test examples =', testX.shape)
print('Dataset: training examples =', trainy.shape, 'test examples =', testy.shape)

#벡터가 unit이 될 때까지 scale하기. 이거 여기서말고 임베딩할때 추가하자
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# 각 유명인 이름 변수 문자열을 정수로 변환
# that means ali, aqsa, manaal, umair will turn into 0, 1, 2, 3
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# 모델에 svc사용.  추측을 하고 나서의 확률을 나중에 원할 수도 있는데, probability를 True로 설정하면 됨
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# 모델 평가하기

# 예측
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# 정확도 점수
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

print('Accuracy: train = %.3f, test = %.3f' %(score_train*100, score_test*100))

# 출력 예시
# 데이터셋: 훈련 93개, 테스트 25개
# 정확도: 훈련=100.000, 테스트=100.000


# 테스트 데이터셋의 얼굴을 불러오기
# 테스트 데이터셋에서 임의의 예제에 대한 테스트 모델
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# 얼굴 임베딩을 입력으로 사용해서 적합한 모델로 추측
# 클래스 정수 번호와 추측 정확도 ㅇㅖ상
samples = random_face_emb.reshape(-1, random_face_emb.shape[0])
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# 예측한 클래스의 정수 번호를 통해 이름을 얻을 수 있고, 이 추측의 정확도를 얻기 가능
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
