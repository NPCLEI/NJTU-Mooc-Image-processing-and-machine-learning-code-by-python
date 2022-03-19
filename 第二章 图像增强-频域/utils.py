import numpy as np
import cv2
import matplotlib.pyplot as plt

def read(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def readGray(path,loss = -1):
    img = cv2.imread(path, 0)

    if loss < 0:
        return img
    else:
        return img * loss
def toInt(bytstr):
    return int.from_bytes(bytstr,byteorder='little', signed=False)

def readRaw(path):
    with open(path,'rb') as f:
        w,h = f.read(4),f.read(4)
        w,h = toInt(w),toInt(h)

        res = np.frombuffer(f.read(),dtype=np.uint8)
        res = res.reshape((h,w))
        # print(res.shape)
        return res.copy()

figure_count = 1

def compare(img,funcs,figsize=(20,20)):
    global figure_count
    plt.figure(figure_count,figsize=figsize)
    figure_count+=1
    fl = len(funcs)
    for fi,func in zip(range(1,fl+1),funcs):
        plt.subplot(1,fl,fi)
        plt.imshow(func(img.copy()),cmap='gray')
