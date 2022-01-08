from typing import TextIO

from PIL import Image
from pytesseract import *

a='HY센스L'
b='HY태백B'
c='양재깨비체B'
d='양재샤넬체'
e='한컴 바겐세일 B'
y='한컴 백제 B'
g='한컴 소망 B'
h='한컴 윤체 M'
i='한컴돋움'
j='휴먼옛체'

r=[a,b,c,d,e,y,g,h,i,j]
for x in range(len(r)):
    filename = "D:/OcrData2/"+r[x]+"_1.jpg"
    image = Image.open(filename)
    text = image_to_string(image, lang="kor")


    with open("D:/OcrData2/"+r[x]+".txt", "w") as f:
        f.write(text)


