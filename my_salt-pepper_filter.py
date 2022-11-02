"""
    Author : Cervoliu
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def PSNR(f : np.ndarray, g : np.ndarray) -> float:
    m, n = f.shape
    mse = 0.0
    for i in range(m):
        for j in range(n):
            mse += (int(f[i][j]) - int(g[i][j])) ** 2
    mse /= (m * n)
    return 10 * math.log10((255 * 255) / mse)

def SSIM(f : np.ndarray, g : np.ndarray) -> float:
    m, n = f.shape
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = f.mean()
    sigma_x = f.std()
    mu_y = g.mean()
    sigma_y = g.std()
    sigma_xy = 0
    for i in range(m):
        for j in range(n):
            sigma_xy += (int(f[i][j]) - mu_x) * (int(g[i][j]) - mu_y)
    sigma_xy /= (m * n - 1)
    return (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / ((mu_x ** 2) + (mu_y ** 2) + c1) / ((sigma_x ** 2) + (sigma_y ** 2) + c2)

def median_filter(img : np.ndarray, ksize : int) -> np.ndarray:
    m, n = img.shape
    res = img.copy()
    for i in range(m):
        for j in range(n):
            if 0 < img[i][j] < 255:
                res[i][j] = img[i][j]
                continue
            arr = []
            for x in range(i - ksize, i + ksize + 1):
                for y in range(j - ksize, j + ksize + 1):
                    if 0 <= x < m and 0 <= y < n:
                        arr.append(img[x][y])
            res[i][j] = np.median(arr)
    return res

if __name__ == "__main__":
    for SNR in range(5, 100, 10):
        pics_dir = ".\\grayscale\\pics\\"
        noise_dir = ".\\grayscale\\addnoise\\salt-pepper_noise\\" + str(SNR) + "\\"
        # filter_dir = ".\\grayscale\\filter\\salt-pepper_filter\\" + str(SNR) + "\\"
        filter_dir = ".\\grayscale\\myfilter\\salt-pepper_filter\\" + str(SNR) + "\\"
        Path(filter_dir).mkdir(parents=True, exist_ok=True)
        result = open(filter_dir + "result.txt", mode='w')

        for file_name in os.listdir(noise_dir):
            if file_name != "lena.bmp":
                continue
            print('SNR = {0}, filename = {1}'.format(SNR, file_name))
            img0 = cv2.imread(pics_dir + file_name, 0)
            img_noise = cv2.imread(noise_dir + file_name, 0)

            ksize_best = 0
            psnr_best = 0
            ssim_best = 0
            img_best = None
            for k in range(3, 16, 2):
                # img_median = cv2.medianBlur(img_noise, k)
                img_median = median_filter(img_noise, k)
                psnr = PSNR(img0, img_median)
                ssim = SSIM(img0, img_median)
                if psnr > psnr_best:
                    psnr_best = psnr
                    ssim_best = ssim
                    ksize_best = k
                    img_best = img_median
                else:
                    break #PSNR is single peak function
            result.write('filename = {0}, best ksize = {1}, psnr = {2}, ssim = {3}\n'.format(
                        file_name, ksize_best, psnr_best, ssim_best))
            cv2.imwrite(filter_dir + file_name, img_best)
