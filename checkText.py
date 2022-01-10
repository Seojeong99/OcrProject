file = open('./phd08_input_samples/가.txt','r')
x = file.read().split()


count = 0
for i in range(len(x)):
    for j in range(len(x[i])):
        if x[i][j] == 'S':
            count = count+1

print(count)
file.close()
