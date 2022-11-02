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

if __name__ == "__main__":
    pics_dir = ".\\grayscale\\pics\\"
    noise_dir = ".\\grayscale\\addnoise\\Gaussian_noise\\" + "\\"
    filter_dir = ".\\grayscale\\filter\\Gaussian_filter\\" + "\\"
    Path(filter_dir).mkdir(parents=True, exist_ok=True)
    result = open(filter_dir + "result.txt", mode='w')

    for file_name in os.listdir(noise_dir):
        print('filename = {0}'.format(file_name))
        img0 = cv2.imread(pics_dir + file_name, 0)
        img_noise = cv2.imread(noise_dir + file_name, 0)

        img_blur = cv2.GaussianBlur(img_noise, (9, 9), 0)

        psnr = PSNR(img0, img_blur)
        ssim = SSIM(img0, img_blur)

        cv2.imwrite(filter_dir + file_name, img_blur)
        result.write('filename = {0}, psnr = {1}, ssim = {2}\n'.format(file_name, psnr, ssim))
