from PIL import Image
from pytesseract import *

filename = "D:\OcrData\펜흘림체_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\펜흘림체.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\한컴 고딕_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\한컴 고딕.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\한컴 솔잎 B체_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\한컴 솔잎 B체.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\한컴 쿨재즈B_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\한컴 쿨재즈B.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\함초롬돋움_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\함초롬돋움.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\함초롱바탕_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\함초롱바탕.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\휴먼가는샘체_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\휴먼가는샘체.txt", "w") as f:
    f.write(text)

filename = "D:\OcrData\휴먼엑스포체_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

with open("D:\OcrData\휴먼엑스포체.txt", "w") as f:
    f.write(text)
