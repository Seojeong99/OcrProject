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

f1 = open("D:\OcrData\origin가는kor.txt","r",encoding='UTF8')
t = f1.read()
thin = list(t)
thin = ' '.join(thin).split()
print("가는kor")
print(jaccard_similarity(original,thin))

f2 = open("D:\OcrData\origin가는kor2.txt","r",encoding='UTF8')
hm = f2.read()
human = list(hm)
human = ' '.join(human).split()
print("가는kor2")
print(jaccard_similarity(original,human))









