# POCTV (intelligence CCTV)
포스코 AI Bigdata 11기 B1조 입니다 :)

### Project Preview
![ㅆ](https://user-images.githubusercontent.com/57425658/102691314-d9916980-424e-11eb-95ec-08ac44af0f83.png)
## data referece and example 
![image](https://user-images.githubusercontent.com/57425658/102691358-3856e300-424f-11eb-8001-97793411a0d1.png)

#### situation list
 - V = Violence(폭력상황)
 - F = Faint(실신상황)
 - NV = Non Violence(정상상황)

### YOLO training result
![image](https://user-images.githubusercontent.com/57425658/102692435-802d3880-4256-11eb-9fde-fb7e07fad222.png)

### Violence situation detect 
![violence](https://github.com/lucas-korea/AI_Bigdata_11th_B1/blob/master/ezgif-3-4f52388ae637.gif)

### faint situation detect 
![faint](https://github.com/lucas-korea/AI_Bigdata_11th_B1/blob/master/ezgif.com-gif-maker%20(2).gif)


이상상황 감지(abnormal situation detector) 
: yolov5를 이용하여 폭력상황, 실신 상황, 정상상황을 학습시켜서 각 상황을 인식합니다

deepsort_v5
: 미리 주어진 사람의 사진 데이터를 기반, deepsort 알고리즘과 yolov5를 이용하여 cctv에 보이는 사람이 누구인지 인식하고 동선을 추적합니다

GUI
: 이상상황 감지와 deepsort에서 인식된 사람의 동선과 상황의 위치를 추적하여 지도에 표시합니다

posco facenet
: keras의 facenet을 이용하여 입구에서 보안 인증을 담당합니다
