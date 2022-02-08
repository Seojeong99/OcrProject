#S를 243번 센 다음에 그 앞까지 잘라서 다시 저장하기
import os
file_list = os.listdir('C:\\Users\\이서정\\Desktop\\한글사진\\')
print(file_list)
for i in range(len(file_list)):
    file = open('C:\\Users\\이서정\\Desktop\\한글사진\\'+file_list[i],'r')
    x = file.read()