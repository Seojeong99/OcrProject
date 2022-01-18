from glob import glob
image_datas = glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*\\*.png')
class_datas = glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*')
txtN = len(glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*'))
class_name = []

for i in range(txtN):
    class_name.append(class_datas[i][59])

for j in range(len(class_name)):
    dic = {class_name[j]:j}
    print(dic)
