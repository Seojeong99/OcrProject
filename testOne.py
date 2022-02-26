from PIL import Image
from pytesseract import *


filename = "D:\OcrData\한컴 쿨재즈B_1.JPG"
#filename = "D:\OcrData\가는안상수체_1.JPG"
#filename = "D:\OcrData2\양재샤넬체_1.JPG"
#filename = "D:\OcrData2\가나다라마바사아자차카타파하_1.JPG"
image = Image.open(filename)

text = image_to_string(image, lang="Hancomff")

print(text)