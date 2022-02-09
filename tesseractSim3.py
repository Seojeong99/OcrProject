from math import *


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality / float(union_cardinality)

f1 = open("C:\\Users\\이서정\\Desktop\\POLARIS LOC\\한글\\sim\\애국가.txt","r",encoding='UTF8')

origin = f1.read()
original = list(origin)
original = ' '.join(original).split()
#print(original)

f1 = open("C:\\Users\\이서정\\Desktop\\POLARIS LOC\\한글\\sim\\test1.txt","r",encoding='UTF8')
t = f1.read()
thin = list(t)
thin = ' '.join(thin).split()
print("1")
print(jaccard_similarity(original,thin))

f2 = open("C:\\Users\\이서정\\Desktop\\POLARIS LOC\\한글\\sim\\test2.txt","r",encoding='UTF8')
hm = f2.read()
human = list(hm)
human = ' '.join(human).split()
print("2")
print(jaccard_similarity(original,human))

f3 = open("C:\\Users\\이서정\\Desktop\\POLARIS LOC\\한글\\sim\\test3.txt","r",encoding='UTF8')
ab = f3.read()
abc = list(ab)
abc = ' '.join(abc).split()
print("3")
print(jaccard_similarity(original,abc))






