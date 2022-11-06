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

def median_filter(img : np.ndarray) -> np.ndarray:
    m, n = img.shape

    check = lambda x, y : 0 <= x < m and 0 <= y < n and 0 < img[x][y] < 255

    res = img.copy()
    for i in range(m):
        for j in range(n):
            if 0 < img[i][j] < 255:
                res[i][j] = img[i][j]
                continue
            arr = []
            for k in range(1, 100): # ksize = 2 * k + 1
                for y in range(j - k, j + k):
                    if check(i - k, y): arr.append(img[i - k][y])
                    if check(i + k, y): arr.append(img[i + k][y])
                for x in range(i - k + 1, i + k):
                    if check(x, j - k): arr.append(img[x][j - k])
                    if check(x, j + k): arr.append(img[x][j + k])
                if check(i - k, j + k): arr.append(img[i - k][j + k])
                if check(i + k, j + k): arr.append(img[i + k][j + k])
                
                if len(arr) > 0: break
            res[i][j] = np.median(arr)
    return res

if __name__ == "__main__":
    for SNR in range(95, 100, 10):
        pics_dir = ".\\grayscale\\pics\\"
        noise_dir = ".\\grayscale\\addnoise\\salt-pepper_noise\\" + str(SNR) + "\\"
        # filter_dir = ".\\grayscale\\filter\\salt-pepper_filter\\" + str(SNR) + "\\"
        filter_dir = ".\\grayscale\\newfilter\\salt-pepper_filter\\" + str(SNR) + "\\"
        Path(filter_dir).mkdir(parents=True, exist_ok=True)
        result = open(filter_dir + "result.txt", mode='w')

        for file_name in os.listdir(noise_dir):
            if file_name != "lena.bmp":
                continue
            print('SNR = {0}, filename = {1}'.format(SNR, file_name))
            img0 = cv2.imread(pics_dir + file_name, 0)
            img_noise = cv2.imread(noise_dir + file_name, 0)

            img_median = median_filter(img_noise)
            psnr = PSNR(img0, img_median)
            ssim = SSIM(img0, img_median)
            result.write('filename = {0}, psnr = {1}, ssim = {2}\n'.format(file_name, psnr, ssim))
            cv2.imwrite(filter_dir + file_name, img_median)
