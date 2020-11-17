
# 로그 파일 frame by frame 읽기
f = open("C:\\Users\\jcy37\\OneDrive\\바탕 화면\\output\\bm_weight_데이터 축소전 100ㅕㅍ체ㄴ\\새 폴더\\faint_two_people.txt", 'r')
data = f.read()
data_splited = data.split('\n\n')
# print(data_splited)
data_splited_obj = []
data_splited_obj_ap = []
print(len(data_splited))

for j in range(len(data_splited)-1): # frame
    data_splited_obj = data_splited[j].split('\n')
    # print('=======================')
    # print(data_splited_obj)
    obj_label = []
    print(j)
    for a in range(len(data_splited_obj)): # of obj
        print(data_splited_obj[0], a)
        # print(a)
        # print((data_splited_obj[a]))

        if(data_splited_obj[0] == 'None'):
            # print('None')
            obj_label.append('None')
        else:
            # data_splited_obj = data_splited_obj[a]
            # print(data_splited_obj)
            # data_splited_obj = data_splited_obj.split(' ')
            # print(data_splited_obj)
            # # data_splited_obj = data_splited_obj.split(' ')
            # obj_label.append(data_splited_obj[0])
            # obj_label.append(data_splited_obj[1])
            # obj_label.append(data_splited_obj[2])
            # ===========================================
            # print(data_splited_obj[a][0])
            # print(data_splited_obj,a)
            data_splited_obj_dummy = data_splited_obj[a].split(' ')
            obj_label.append(data_splited_obj_dummy[0])
            obj_label.append(data_splited_obj_dummy[1])
            obj_label.append(data_splited_obj_dummy[2])
    data_splited_obj_ap.append(obj_label)
print(data_splited_obj_ap)

# print('===========')
# print(data_splited_obj_ap)
#
# print(data_splited_obj_ap[0])
# print(data_splited_obj_ap[0][0])
# print(len(data_splited_obj_ap))
# print(data_splited_obj_ap[10])


# 판단 시작
faint_flag = 0
faint_frame = 0
faint_stack  = 0
violence_flag = 0
violence_frame = 0
violence_stack = 0

num0 = '0'
num1 = '1'
num2 = '2'
for i in range(len(data_splited_obj_ap)):
    if num1 in data_splited_obj_ap[i] and violence_flag < 2: # 상황이 일어났는지 보고, 만약 real 상황으로 판단하면 더이상 판단하지 않음
        # print('V!')
        # print(violence_frame , i)
        if violence_flag == 0:
             violence_frame = i
        violence_flag = 1 # unsure violence_상황
        violence_stack +=1
        if violence_stack >= 20:
            violence_flag = 2 # real violence 상황으로 판단
        if (i - violence_frame) > 100: # 100 frame 내로 상황 판단 못할시 strack 데이터 무시
            violence_stack = 0
            violence_flag = 0
    if num2 in data_splited_obj_ap[i] and faint_flag < 2: # 상황이 일어났는지 보고, 만약 real 상황으로 판단하면 더이상 판단하지 않음
        # print('faint!')
        if faint_flag == 0:
             faint_frame = i
        faint_flag = 1 # unsure violence_상황
        faint_stack +=1
        if faint_stack >= 20:
            faint_flag = 2 # real violence 상황으로 판단
        if (i - faint_frame) > 100: # 100 frame 내로 상황 판단 못할시 strack 데이터 무시
            faint_stack = 0
            faint_flag = 0
    # print('frame: ', i, 'faint flag   :', faint_flag, '  faint stack    :', faint_stack)
    # print('frame: ', i, 'violence_flag:', violence_flag, '  violence_stack :', violence_stack)



    # if faint_flag == 2:
    #     print(i,"faint emergency !!!!!!!!!!!!!!")
    # if violence_flag == 2:
    #     print(i,"violence emergency !!!!!!!!!!!!!!")


# for i in range(len(frame)):
#     if violence_detect_frame and violence_flag < 2: #real flag 세워지면 더이상 판단 안함
#         if violence_flag = 0:
#              violence_frame = frame
#
#
#         if violence_stack => 20:
#             violence_falg = 2 # real violence 상황으로 판단
#         else:
#             violence_flag = 1  # unsure violence_상황
#         violence_stack += 1
#         if (frame - violence_frame) >= 100 frame

f.close()