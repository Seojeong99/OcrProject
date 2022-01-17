from typing import TextIO

from PIL import Image
from pytesseract import *


filename = "D:\\project\\W\\W_1.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="eng")


with open("D:\\project\\W\\Wheaton1997_Article_TheCyclicBehaviorOfTheGreaterL-1.txt", "w",encoding='UTF8') as f:
    print(f.write(text))

filename = "D:\\project\\W\\W_15.jpg"
image = Image.open(filename)
text = image_to_string(image, lang="eng")

with open("D:\\project\\W\\Wheaton1997_Article_TheCyclicBehaviorOfTheGreaterL-15.txt", "w",encoding='UTF8') as f:
    print(f.write(text))


