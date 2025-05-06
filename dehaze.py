import cv2 as cv
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

def DarkChannel(img, size):
    """
        計算影像的 Dark Channel
    """
    b, g, r = cv.split(img)
    dc = cv.min(cv.min(r, g), b) 
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    darkC = cv.erode(dc, kernel)
    
    return darkC

def AtmLight(img, dark):
    """
        計算大氣光照明強度
    """
    [h, w] = img.shape[:2]
    imgsize = h * w
    numpx = int(max(math.floor(imgsize / 1000), 1))
    darkvec = dark.reshape(imgsize)
    imvec = img.reshape(imgsize, 3)

    indices = darkvec.argsort()
    indices = indices[imgsize - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(img, A, size):
    """
        計算 transmission map(透射率圖)
    """
    omega = 0.95
    img3 = np.empty(img.shape, img.dtype)

    for ind in range(0, 3):
        img3[:, :, ind] = img[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(img3, size)
    return transmission


def Guidedfilter(img, p, ksize, eps):
    """
        應用 guided filter 進行細化
    """
    mean_I = cv.boxFilter(img, cv.CV_64F, (ksize, ksize))
    mean_p = cv.boxFilter(p, cv.CV_64F, (ksize, ksize))
    mean_Ip = cv.boxFilter(img * p, cv.CV_64F, (ksize, ksize))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv.boxFilter(img * img, cv.CV_64F, (ksize, ksize))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv.boxFilter(a, cv.CV_64F, (ksize, ksize))
    mean_b = cv.boxFilter(b, cv.CV_64F, (ksize, ksize))

    q = mean_a * img + mean_b
    
    return q


def TransmissionRefine(img, teMap):
    """
        細化 transmission map(透射率圖)
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    ksize = 60
    eps = 0.0001
    transmissionMap = Guidedfilter(gray, teMap, ksize, eps)

    return transmissionMap


def Recover(img, tMap, A, t0=0.1):
    """
        重建除霧後的影像
    """
    res = np.empty(img.shape, img.dtype)
    tMap = cv.max(tMap, t0)

    for ind in range(0, 3):
        res[:, :, ind] = (img[:, :, ind] - A[0, ind]) / tMap + A[0, ind]

    return res


if __name__ == '__main__':
    input_image_path = './input_image/'
    output_image_path = './output_image/'
    tmp_path = './tmp/'
    fn = 'hazy08.jpg'
    src = cv.imread(input_image_path + fn)

    I = src.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    teMap = TransmissionEstimate(I, A, 15)
    RefineMap = TransmissionRefine(src, teMap)
    DeHazeImg = Recover(I, RefineMap, A, 0.1)

    I = (I * 255).astype('uint8')
    dark = (dark * 255).astype('uint8')
    teMap = (teMap * 255).astype('uint8')
    RefineMap = (RefineMap * 255).astype('uint8')
    DeHazeImg = (DeHazeImg * 255).astype('uint8')

    psnr = PSNR(I, DeHazeImg)
    ssim = SSIM(I, DeHazeImg, data_range = I.max() - I.min(), win_size = 3, multichannel = True)

    print(f'psnr = {psnr}, ssim = {ssim}')
    
    window_width = 800
    window_height = window_width * dark.shape[0] // dark.shape[1]

    cv.namedWindow('Source image', cv.WINDOW_NORMAL)
    cv.namedWindow('dark', cv.WINDOW_NORMAL)
    cv.namedWindow('teMap', cv.WINDOW_NORMAL)
    cv.namedWindow('RefineMap', cv.WINDOW_NORMAL)
    cv.namedWindow('DeHazeImg', cv.WINDOW_NORMAL)
    
    cv.imshow('Source image', src)
    cv.imshow('dark', dark)
    cv.imshow("teMap", teMap)
    cv.imshow("RefineMap", RefineMap)
    cv.imshow('DeHazeImg', DeHazeImg)

    cv.resizeWindow('Source image', window_width, window_height)
    cv.resizeWindow('dark', window_width, window_height)
    cv.resizeWindow('teMap', window_width, window_height)
    cv.resizeWindow('RefineMap', window_width, window_height)
    cv.resizeWindow('DeHazeImg', window_width, window_height)

    cv.imwrite(tmp_path + 'drak'+ fn, dark)
    cv.imwrite(tmp_path + 'teMap' + fn, teMap)
    cv.imwrite(tmp_path + 'reMap' + fn, RefineMap)
    cv.imwrite(output_image_path + fn, DeHazeImg)

    cv.waitKey()