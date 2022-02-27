#S를 243번 센 다음에 그 앞까지 잘라서 다시 저장하기
import os
file_list = os.listdir('C:\\Users\\이서정\\Desktop\\한글사진\\')
print(file_list)

for q in range(len(file_list)):
    file = open('C:\\Users\\이서정\\Desktop\\한글사진\\'+file_list[q],'r')
    x = file.read()

    count = 0

    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 'S':
                count = count + 1
                if count == 52:  # -2개만큼 저장
                    with open('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_input_samples\\'+file_list[q], 'w') as f:
                        for z in range(i - 1):
                            f.writelines(x[z])




print(count)
file.close()