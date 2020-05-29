import skimage
import random
import scipy
import numpy
import numba
import math
import time
import cv2
import os
from numba import jit
from scipy import ndimage
from skimage.util import img_as_ubyte
dirpath = r'D:\u_data\2020-04-23-164152\originimages'
savepath = r'D:\u_data\2020-04-23-164152\saveimages'
imagelist = os.listdir(dirpath)


def RemoveFogByGlobalHisteq():
    for imagename in imagelist:
        fullpath = os.path.join(dirpath, imagename)
        image = cv2.imread(fullpath, cv2.IMREAD_COLOR)
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # (R,G,B)=cv2.split(image)
        (B, G, R) = cv2.split(image)
        M = cv2.equalizeHist(R)
        N = cv2.equalizeHist(G)
        L = cv2.equalizeHist(B)
        result = cv2.merge([L, N, M])
        cv2.imwrite(os.path.join(savepath, imagename), result)


# RemoveFogByGlobalHisteq()


def RemoveFogByLocalHisteq():
    for imagename in imagelist:
        fullpath = os.path.join(dirpath, imagename)
        image = cv2.imread(fullpath, cv2.IMREAD_COLOR)
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # (R,G,B)=cv2.split(image)
        (B, G, R) = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
        M = clahe.apply(R)
        N = clahe.apply(G)
        L = clahe.apply(B)
        result = cv2.merge([L, N, M])
        cv2.imwrite(os.path.join(savepath, imagename), result)


# RemoveFogByLocalHisteq()


@jit
def RemoveFogByRetinex():
    for imagename in imagelist:
        fullpath = os.path.join(dirpath, imagename)
        image = cv2.imread(fullpath, cv2.IMREAD_COLOR)
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # (R,G,B)=cv2.split(image)
        (B, G, R) = cv2.split(image)
        # Gray = R*0.299 + G*0.587 + B*0.114
        # Gray = (R*299 + G*587 + B*114 + 500) / 1000
        # Gray = (R*30 + G*59 + B*11 + 50) / 100
        # Gray = (R + (WORD)G<<1 + B) >> 2
        # Gray = (R*1 + G*2 + B*1) >> 2
        # Gray = (R*2 + G*5 + B*1) >> 3
        # Gray = (R*4 + G*10 + B*2) >> 4
        # Gray = (R*9 + G*19 + B*4) >> 5
        # Gray = (R*19 + G*37 + B*8) >> 6
        # Gray = (R*38 + G*75 + B*15) >> 7
        # Gray = (R*76 + G*150 + B*30) >> 8
        # Gray = (R*153 + G*300 + B*59) >> 9
        # Gray = (R*306 + G*601 + B*117) >> 10
        # Gray = (R*612 + G*1202 + B*234) >> 11
        # Gray = (R*1224 + G*2405 + B*467) >> 12
        # Gray = (R*2449 + G*4809 + B*934) >> 13
        # Gray = (R*4898 + G*9618 + B*1868) >> 14
        # Gray = (R*9797 + G*19235 + B*3736) >> 15
        # Gray = (R*19595 + G*38469 + B*7472) >> 16
        # Gray = (R*39190 + G*76939 + B*14943) >> 17
        # Gray = (R*78381 + G*153878 + B*29885) >> 18
        # Gray = (R*156762 + G*307757 + B*59769) >> 19
        # Gray = (R*313524 + G*615514 + B*119538) >> 20
        # Gray = (R^2.2 * 0.2973 + G^2.2 * 0.6274 + B^2.2 * 0.0753)^(1/2.2)
        # GRAY = (RED+BLUE+GREEN)/3
        print('R:', R)
        fr = numpy.float32(R)
        fg = numpy.float32(G)
        fb = numpy.float32(B)
        mr = cv2.normalize(fr, 0, 1.0, cv2.NORM_MINMAX)
        mg = cv2.normalize(fg, 0, 1.0, cv2.NORM_MINMAX)
        mb = cv2.normalize(fb, 0, 1.0, cv2.NORM_MINMAX)
        alpha = random.randint(80, 100) * 20
        n = math.floor(min(numpy.shape(image)[0], numpy.shape(image)[1]) * 0.5)
        n1 = math.floor((n + 1) / 2)
        b = numpy.zeros(shape=(n, n))
        for i in range(0, n, 1):
            for j in range(0, n, 1):
                b[i, j] = math.exp(-((i - n1) ^ 2 + (j - n1) ^ 2) /
                                   (4 * alpha)) / (math.pi * alpha)
        nr1 = scipy.ndimage.convolve(mr, b, mode='nearest')
        ng1 = scipy.ndimage.convolve(mg, b, mode='nearest')
        nb1 = scipy.ndimage.convolve(mb, b, mode='nearest')
        # nr1 = cv2.filter2D(mr, -1, b, borderType=cv2.BORDER_CONSTANT)
        # ng1 = cv2.filter2D(mg, -1, b, borderType=cv2.BORDER_CONSTANT)
        # nb1 = cv2.filter2D(mb, -1, b, borderType=cv2.BORDER_CONSTANT)
        # nr1 = scipy.ndimage.correlate(mr, b, mode='constant')
        # ng1 = scipy.ndimage.correlate(mg, b, mode='constant')
        # nb1 = scipy.ndimage.correlate(mb, b, mode='constant')
        print('nr1:', nr1)
        ur1 = numpy.log(nr1)
        ug1 = numpy.log(ng1)
        ub1 = numpy.log(nb1)
        tr1 = numpy.log(mr)
        tg1 = numpy.log(mg)
        tb1 = numpy.log(mb)
        yr1 = (tr1 - ur1) / 3
        yg1 = (tg1 - ug1) / 3
        yb1 = (tb1 - ub1) / 3
        beta = random.randint(80, 100) * 1
        x = 32
        a = numpy.zeros(shape=(n, n))
        for i in range(0, n, 1):
            for j in range(0, n, 1):
                a[i, j] = math.exp(-((i - n1) ^ 2 + (j - n1) ^ 2) /
                                   (4 * beta)) / (6 * math.pi * beta)
        nr2 = scipy.ndimage.convolve(mr, a, mode='nearest')
        ng2 = scipy.ndimage.convolve(mg, a, mode='nearest')
        nb2 = scipy.ndimage.convolve(mb, a, mode='nearest')
        # nr2 = cv2.filter2D(mr, -1, a, borderType=cv2.BORDER_CONSTANT)
        # ng2 = cv2.filter2D(mg, -1, a, borderType=cv2.BORDER_CONSTANT)
        # nb2 = cv2.filter2D(mb, -1, a, borderType=cv2.BORDER_CONSTANT)
        # nr2 = scipy.ndimage.correlate(mr, a, mode='constant')
        # ng2 = scipy.ndimage.correlate(mg, a, mode='constant')
        # nb2 = scipy.ndimage.correlate(mb, a, mode='constant')
        print('nr2:', nr2)
        ur2 = numpy.log(nr2)
        ug2 = numpy.log(ng2)
        ub2 = numpy.log(nb2)
        tr2 = numpy.log(mr)
        tg2 = numpy.log(mg)
        tb2 = numpy.log(mb)
        yr2 = (tr2 - ur2) / 3
        yg2 = (tg2 - ug2) / 3
        yb2 = (tb2 - ub2) / 3
        eta = random.randint(80, 100) * 200
        e = numpy.zeros(shape=(n, n))
        for i in range(0, n, 1):
            for j in range(0, n, 1):
                e[i, j] = math.exp(-((i - n1) ^ 2 + (j - n1) ^ 2) /
                                   (4 * eta)) / (4 * math.pi * eta)
        nr3 = scipy.ndimage.convolve(mr, e, mode='nearest')
        ng3 = scipy.ndimage.convolve(mg, e, mode='nearest')
        nb3 = scipy.ndimage.convolve(mb, e, mode='nearest')
        # nr3 = cv2.filter2D(mr, -1, e, borderType=cv2.BORDER_CONSTANT)
        # ng3 = cv2.filter2D(mg, -1, e, borderType=cv2.BORDER_CONSTANT)
        # nb3 = cv2.filter2D(mb, -1, e, borderType=cv2.BORDER_CONSTANT)
        # nr3 = scipy.ndimage.correlate(mr, e, mode='constant')
        # ng3 = scipy.ndimage.correlate(mg, e, mode='constant')
        # nb3 = scipy.ndimage.correlate(mb, e, mode='constant')
        print('nr3:', nr3)
        ur3 = numpy.log(nr3)
        ug3 = numpy.log(ng3)
        ub3 = numpy.log(nb3)
        tr3 = numpy.log(mr)
        tg3 = numpy.log(mg)
        tb3 = numpy.log(mb)
        yr3 = (tr3 - ur3) / 3
        yg3 = (tg3 - ug3) / 3
        yb3 = (tb3 - ub3) / 3
        dr = yr1 + yr2 + yr3
        dg = yg1 + yg2 + yg3
        db = yb1 + yb2 + yb3
        print('dr:', dr)
        cr = img_as_ubyte(cv2.convertScaleAbs(dr * 180))
        cg = img_as_ubyte(cv2.convertScaleAbs(dg * 180))
        cb = img_as_ubyte(cv2.convertScaleAbs(db * 180))
        print('cr:', cr)
        result = cv2.merge([cb, cg, cr])
        print('result:', result)
        cv2.imwrite(os.path.join(savepath, imagename), result)


# kernel = np.ones((3, 3), np.float32)/9
# grayFrame = cv2.filter2D(SrcgrayFrame.astype('float32'), -1, kernel,borderType=cv2.BORDER_CONSTANT)
t0 = time.time()
RemoveFogByRetinex()
t1 = time.time()
print(t1 - t0)
