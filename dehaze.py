import cv2 as cv
import numpy as np
import math

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

def multi_scale_retinex(img, scales, weights):
    """
        多尺度 Retinex 演算法
    """
    img_log = np.log1p(img.astype(np.float32)) 

    msr = np.zeros_like(img, dtype=np.float32)

    for i, scale in enumerate(scales):
        # 將 scale 轉為正奇數
        ksize = int(scale)
        if ksize % 2 == 0:
            ksize += 1
        if ksize <= 0:
            ksize = 1

        img_blur = cv.GaussianBlur(img_log, (ksize, ksize), 0)

        retinex = img_log - np.log1p(img_blur)

        # 加權累加結果
        msr += weights[i] * retinex

    # 正規化結果
    msr = cv.normalize(msr, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return msr

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
        img3[:, :, ind] = img[:, :, ind] / A[0, ind] # 將影像除以大氣光照明強度 

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
        res[:, :, ind] = (img[:, :, ind] - A[0, ind]) / tMap + A[0, ind] # 將影像除以大氣光照明強度

    return res

def white_balance(img):
    """
        白平衡
    """

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    img_balanced = img.copy()
    img_balanced[:, :, 0] = np.clip(img[:, :, 0] * avg_gray / avg_b, 0, 255)
    img_balanced[:, :, 1] = np.clip(img[:, :, 1] * avg_gray / avg_g, 0, 255)
    img_balanced[:, :, 2] = np.clip(img[:, :, 2] * avg_gray / avg_r, 0, 255)

    return img_balanced.astype(np.uint8) # 將影像轉換回 uint8 格式

# 調整亮度

def increase_brightness_hsv(img, factor=1.2):
    """
        使用 HSV 模式調整亮度
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    hsv = cv.merge((h, s, v))

    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# def apply_clahe(img):
#     """
#         對影像亮度通道使用 CLAHE 增亮
#     """
#     lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
#     l, a, b = cv.split(lab)

#     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     lab = cv.merge((l, a, b))
#     return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

if __name__ == '__main__':
    input_image_path = './input_image/'
    output_image_path = './output_image/'
    tmp_path = './tmp/'
    
    for i in range(1, 10):
        fn = 'hazy0'+ str(i) + '.jpg'
        src = cv.imread(input_image_path + fn)

        # 設定 MSR 參數
        scales = [15, 80, 250]  # 高斯濾波器大小
        weights = [1/3, 1/3, 1/3]  # 尺度權重
        
        # 應用 MSR
        dehazed_img_msr = multi_scale_retinex(src.astype(np.float32), scales, weights)
        
        I = dehazed_img_msr.astype('float64') / 255
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
        DeHazeImg = white_balance(DeHazeImg)
        # DeHazeImg = color_correction_lab(DeHazeImg)
        DeHazeImg = increase_brightness_hsv(DeHazeImg, 1.2)
        # DeHazeImg = apply_clahe(DeHazeImg)

        cv.imwrite(tmp_path + 'drak'+ fn, dark)
        cv.imwrite(tmp_path + 'teMap' + fn, teMap)
        cv.imwrite(tmp_path + 'reMap' + fn, RefineMap)
        cv.imwrite(output_image_path + fn, DeHazeImg)

        print("第" + str(i) + "張影像處理完成")
        
        cv.waitKey()