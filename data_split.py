import sys
import numpy as np 
import os
import random

data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24],[25,45,56],[44,33,43]]
random.shuffle(data)
train = data[0:4]
test = data[5:8]

print(train)