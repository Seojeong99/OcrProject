from PIL import Image
from glob import glob
import os

def convert_png_to_jpg(path):
    # jpg파일을 저장하기 위한 디렉토리의 생성
    if not os.path.exists(path+'_jpg'):
        os.mkdir(path+'_jpg')

    # 모든 png 파일의 절대경로를 저장
    all_image_files=glob(path+'/*.png')

    for file_path in all_image_files:                   # 모든 png파일 경로에 대하여
        img = Image.open(file_path).convert('RGB')  # 이미지를 불러온다.

        directories=file_path.split('\\')                # 절대경로상의 모든 디렉토리를 얻어낸다.
        directories[-2]+='_jpg'                     # 저장될 디렉토리의 이름 지정
        directories[-1]=directories[-1][:-4]+'.jpg'  # 저장될 파일의 이름 지정
        save_filepath='\\'.join(directories)          # 절대경로명으로 바꾸기
        img.save(save_filepath, quality=100)       #


class_datas = glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*')
txtN = len(glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*'))
class_name = []

for i in range(txtN):
    class_name.append(class_datas[i][59])

for i in range (txtN):
    path='C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\'+class_name[i]
    convert_png_to_jpg(path)