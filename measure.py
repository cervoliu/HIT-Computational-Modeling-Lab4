"""
    This program calculates PSNR and SSIM values.
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
    original_dir = ".\\pics\\grayscale\\"
    filter_dir = ".\\filter\\grayscale\\salt-pepper_filter\\"
    for file_name in os.listdir(original_dir):
        img0 = cv2.imread(original_dir + file_name, 0)
        img1 = cv2.imread(filter_dir + file_name, 0)
        