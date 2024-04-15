import scipy.io
import pandas as pd
import numpy as np

# Turn .mat file into .csv

mat = scipy.io.loadmat('imagelabels.mat')
# print(type(mat))
# for key, value in mat.items() :
#     print (key, value)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

print(mat['labels'][0], len(mat['labels'][0]))
print(mat['labels'][0][0])

that_was_hard = []

for val in range(0,8189):
    that_was_hard.append([str((val+1)), str((mat['labels'][0][val]))])

print(that_was_hard[0])

a = np.asarray(that_was_hard)
print (a[0:10])
np.savetxt("example.csv", a, delimiter=",", fmt='%s')

