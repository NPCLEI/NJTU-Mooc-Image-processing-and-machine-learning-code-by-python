from turtle import hideturtle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def distMat(img,center = (0,0),sqrt = True):
    ch,cw = center
    h,w = img.shape
    res = np.array([[(wi - cw) ** 2 + (hi - ch) ** 2 for wi in range(w)] for hi in range(h)])
    if sqrt:
        return np.power(res,0.5)
    else:
        return res

def drawCircleToMat(img,r = 1 ,center=(-1,-1),clone = True):
    if clone:
        img = img.copy()
    h,w = img.shape
    x,y = center
    if center == (-1,-1):
        x,y = (w//2,h//2)

    maxv = np.max(img)

    wi = np.array([i for i in range(x-r,x+r+1)])
    hi_up =  (y + np.sqrt(r**2 - (wi-x)**2)).astype(np.int)
    hi_down = (y - np.sqrt(r**2 - (wi-x)**2)).astype(np.int)
    img[hi_up,wi] = maxv
    img[hi_down,wi] = maxv

    return img

def circleMat(img,r = 1 ,center=(-1,-1),fill = False):
    h,w = img.shape
    x,y = center
    if center == (-1,-1):
        x,y = (h//2,w//2)
    res = np.zeros_like(img)
    wi = np.array([i for i in range(x-r,x+r+1)])
    hi_up =  (y + np.sqrt(r**2 - (wi-x)**2)).astype(np.int)
    hi_down = (y - np.sqrt(r**2 - (wi-x)**2)).astype(np.int)
    res[hi_up,wi] = 1
    res[hi_down,wi] = 1

    return res

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

def Normalization(mtx):
    a,i = np.max(mtx),np.min(mtx)

    return (mtx - i) / (a - i)

if __name__ == "__main__":
    print(distMat(np.zeros((5,5))))
    