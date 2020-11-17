# AI_Bigdata_11th_B1
포스코 AI Bigdata 11기 B1조 입니다 :)

우리는 지능형 CCTV 개발을 목표로 하고 있습니다

저희 내용을 보고 싶으면 powerpoint를 봐보세요

조장 : 정찬영
노태현
황다빈
정종훈
김동현
김보민

이상상황 감지 
: yolov5를 이용하여 폭력상황, 실신 상황, 정상상황을 학습시켜서 각 상황을 인식합니다

deepsort_v5
: 미리 주어진 사람의 사진 데이터를 기반, deepsort 알고리즘과 yolov5를 이용하여 cctv에 보이는 사람이 누구인지 인식하고 동선을 추적합니다

GUI
: 이상상황 감지와 deepsort에서 인식된 사람의 동선과 상황의 위치를 추적하여 지도에 표시합니다

posco facenet
: keras의 facenet을 이용하여 입구에서 보안 인증을 담당합니다