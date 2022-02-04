from PIL import Image
from pytesseract import *

filename = 'C:\\Users\\이서정\\Desktop\\POLARIS LOC\\한글\\sim\\한글 띄어쓰기 없는거_1.jpg'
#filename ='D:\\OcrData\\가는안상수체_1.jpg'
image = Image.open(filename)
text1 = image_to_string(image, lang="new+new4")
print(text1)
print('-------------------------\n')


text2 = image_to_string(image, lang="kor")
print(text2)
print('-------------------------\n')

text3 = image_to_string(image, lang="kor+new4")
print(text3)
print('-------------------------\n')

#text3 = image_to_string(image, lang="kor+new")
#print(text3)
#print('-------------------------\n')

#config=('-l new+kor --oem 3 --psm 4')
#text = image_to_string(image, lang="new")
#text=image_to_string(image, config=config)
#print(text)