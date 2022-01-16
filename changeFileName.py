import os  # os모듈 임포트
from glob import glob

class_datas = glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*')
txtN = len(glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*'))

class_name = []

for i in range(txtN):
    class_name.append(class_datas[i][59])

for i in range(txtN):
    old_name="C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\"+class_name[i]+"_jpg"
    new_name="C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\"+class_name[i]
    os.rename(old_name,new_name)