from PIL import Image
from pytesseract import *

filename = "D:\OcrData2\휴먼옛체_1.JPG"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

print(text)