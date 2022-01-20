from PIL import Image
from pytesseract import *

filename = "D:\OcrData\동.JPG"
image = Image.open(filename)
text = image_to_string(image, lang="kor")

print(text)