from math import *


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality / float(union_cardinality)

f1 = open("D:\OcrData\Original.txt","r",encoding='UTF8')
origin = f1.read()
original = list(origin)
original = ' '.join(original).split()
#print(original)

a="휴먼명조체"
b="궁서체"
c="HY목각파임B"
d="가는안상수체"
e="굴림"
f="나눔고딕"
g="돋움"
h="맑은고딕"
i="양재꽃게체"
j="양재난초체M"
k="양재둘기체M"
l="양재블럭체"
m="펜흘림체"
n="한컴 고딕"
o="한컴 솔잎 B체"
p="한컴 쿨재즈B"
q="함초롬돋움"
r="함초롱바탕"
s="휴먼가는샘체"
t="휴먼엑스포체"

f2 = open("D:\OcrData\human.txt","r",encoding='UTF8')
hm = f2.read()
human = list(hm)
human = ' '.join(human).split()
print("휴먼명조체")
print(jaccard_similarity(original,human))

f3 = open("D:\OcrData\Gung.txt","r",encoding='cp949')
Gu = f3.read()
Gung = list(Gu)
Gung = ' '.join(Gung).split()
print("궁서체")
print(jaccard_similarity(original,Gung))

f4 = open("D:/OcrData/"+c+".txt","r",encoding='cp949')
c1 = f4.read()
c2 = list(c1)
c2 = ' '.join(c2).split()
print(c)
print(jaccard_similarity(original,c1))

f4 = open("D:/OcrData/"+c+".txt","r",encoding='cp949')
c1 = f4.read()
c2 = list(c1)
c2 = ' '.join(c2).split()
print(c)
print(jaccard_similarity(original,c1))

r=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t]
for i in range(3,len(r)):
    f4 = open("D:/OcrData/"+r[i]+".txt","r",encoding='cp949')
    c1 = f4.read()
    c2 = list(c1)
    c2 = ' '.join(c2).split()
    print(r[i])
    print(jaccard_similarity(original,c1))







