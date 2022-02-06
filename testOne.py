from PIL import Image
from pytesseract import *

#traineddata ='kor2'
filename = "D:\OcrData\한글1.JPG"
#filename = "D:\OcrData2\가나다라마바사아자차카타파하_1.JPG"
image = Image.open(filename)

text = image_to_string(image, lang="kor")

print(text)