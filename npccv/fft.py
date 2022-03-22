import numpy as np
import math
import utils

import sys
sys.path.append("../")
import npccv.fft as nft
import npccv.utils as utils
import numpy as np

class fftimg:
    """
    承担傅里叶变换后的一些高低通滤波任务
        示例:
            fimg = fftimg(img)
            fimg.highpass(15,type="gauss"),
    """
    def __init__(self,img):
        """
            img:tpye->numpy array,one channel
        """
        self.img = img
        self.fft = nft.FFT2(self.img,depart=False)
        self.stm,self.pse = np.abs(self.fft),np.angle(self.fft)
        self.powerMtx = None
        self.power = None

    def PowerMtx(self):
        pass

    def Power(self):
        """
        图像总功率
        """
        return np.sum(np.abs(self.fft) ** 2)

    def lowpass(self,d0 = 30,type = "ideal",n = 2,setRes = False):
        """
        frequency domain filtering
        频域滤波器
        d0:截止半径
        type:默认理想低通滤波器,其他有"butterworth"、"gauss"
        lowpass:
        setRes:调用函数后是否立即改变对象内置的原始图像,默认为False
        """
        huv = np.zeros_like(self.img)
        h,w = self.img.shape

        import math
        if type == "gauss":
            dm = utils.distMat(self.img,(h//2,w//2),sqrt=False)
            huv = np.power(math.e,-1 * dm / (2*d0**2) )
        elif type == "butterworth":
            dm = utils.distMat(self.img,(h//2,w//2))
            huv = 1 / (1 + (dm/d0)**(2*n))
        else:
            dm = utils.distMat(self.img,(h//2,w//2))
            huv[np.where(dm <= d0)] = 1
            huv[np.where(dm > d0)] = 0
        
        sf = np.fft.fftshift(self.fft)
        sf = np.fft.fftshift(sf * huv)
        
        res =  nft.iFFT2(sf)

        if setRes:
            self.img = res

        return res

    def highpass(self,d0 = 30,type = "ideal",n = 2,setRes = False):
        """
        frequency domain filtering
        频域滤波器
        d0:截止半径
        type:默认理想低通滤波器,其他有"butterworth"、"gauss"
        lowpass:
        setRes:调用函数后是否立即改变对象内置的原始图像,默认为False
        """
        huv = np.zeros_like(self.img)
        h,w = self.img.shape

        import math
        if type == "gauss":
            dm = utils.distMat(self.img,(h//2,w//2),sqrt=False)
            huv = 1 - np.power(math.e,-1 * dm / (2*d0**2) )
        elif type == "butterworth":
            dm = utils.distMat(self.img,(h//2,w//2))
            huv = 1 / (0.414214+ (d0/dm)**(2*n))
        else:
            dm = utils.distMat(self.img,(h//2,w//2))
            huv[np.where(dm <= d0)] = 0
            huv[np.where(dm > d0)] = 1
        
        sf = np.fft.fftshift(self.fft)
        sf = np.fft.fftshift(sf * huv)

        res =  nft.iFFT2(sf)
        if setRes:
            self.img = res
        return res


    def FSD(self):
        """
            返回频谱,频域,用于绘图
        """
        res = np.fft.fftshift(self.fft)

        stm = np.log(np.abs(res))
        pse = np.angle(res)

        return  stm,pse

def FFT2(img,shift = False,depart = True):
    """
    img:单通道图片
    shift(False):是否将零频域转移到图像中心
    depart(True):是否将结果的实部和虚部分离
    """

    h,w = img.shape
    wMh = np.array([[hi for wi in range(w)] for hi in range(w)])
    wMw = np.array([[wi for wi in range(w)] for hi in range(w)])

    hMh = np.array([[hi for wi in range(h)] for hi in range(h)])
    hMw = np.array([[wi for wi in range(h)] for hi in range(h)])

    c1 = -2j * math.pi 

    G1 = np.power(math.e,c1 * wMw * wMh / w)
    G2 = np.power(math.e,c1 * hMw * hMh / h)

    res = np.dot(np.dot(G2,img),G1)

    if not depart:
        return res

    # print(res)

    #课上讲的
    # real,imag = np.real(res),np.imag(res)
    # stm = (real**2 + imag**2)**0.5
    # pse = np.arctan2(imag / real)

    #github上大神用的
    #先将零频域移到中心,就是将矩阵四角折到中心,注意,PPT上的示例增幅是没有移动的,而频率却移动了
    if shift:
        res = np.fft.fftshift(res)

    stm = np.log(np.abs(res))
    pse = np.angle(res)

    return  stm,pse

def Shift(fft2res):
    pass

def iFFT2(fft2res,returnReal = True):
    """
    fft2res:单通道
    returnReal:是否只返回实部,默认true
    """
    h,w = fft2res.shape
    wMh = np.array([[hi for wi in range(w)] for hi in range(w)])
    wMw = np.array([[wi for wi in range(w)] for hi in range(w)])

    hMh = np.array([[hi for wi in range(h)] for hi in range(h)])
    hMw = np.array([[wi for wi in range(h)] for hi in range(h)])

    c1 = -2j * math.pi 

    G3 = 1/w * np.power(math.e,c1 * wMw * wMh / w)
    G4 = 1/h * np.power(math.e,c1 * hMw * hMh / h)

    fft2res = fft2res.astype(np.complex)
    img = np.dot(np.dot(G4,fft2res),G3)
    if returnReal:
        img = np.abs(img) 
    img = np.rot90(img, 2)

    return  img


if __name__ == "__main__":

    sample = utils.readGray("../imgs/c2/fft.jpg")
    stm,pse = FFT2(sample)
    res = FFT2(sample,depart=False)
    # img = iFFT2(res)
    # res = FFT2(img,depart=False)

    funcs = [lambda img:img,lambda img:stm,lambda img:pse,lambda img:iFFT2(res)]
    utils.compare(sample,funcs)

    stm,pse = FFT2(sample,True)
    res = FFT2(sample,depart=False)
    funcs = [lambda img:img,lambda img:stm,lambda img:pse]
    utils.compare(sample,funcs)
