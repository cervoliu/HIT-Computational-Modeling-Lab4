"""
    Author : Cervoliu
    This program adds noise to orignal images and save to local directory, which is created if not existed.
"""
import os
from pathlib import Path
import numpy as np
import cv2
import random

pepper = 0.5
salt = 1 - pepper

def add_pepper_salt_noise(img : np.ndarray, SNR : float, pepper : float = 0.5) -> np.ndarray:
    """
    Args:
        SNR : Signal-to-Noise Rate in percentage. 0 <= SNR <= 100.
        pepper : Pepper-noise Rate. Defaults to 0.5.
    """
    
    w, h = img.shape
    img_new = img.copy()
    for i in range(w):
        for j in range(h):
            if random.random() > 0.01 * SNR: continue
            if random.random() < pepper:
                img_new[i][j] = 0
            else:
                img_new[i][j] = 255
    return img_new

def add_Gaussian_noise(img : np.ndarray, mean : float = 0, variance : float = 0.1) -> np.ndarray:
    image = np.asarray(img / 255.0, dtype=np.float32)
    noise = np.random.normal(mean, variance, img.shape).astype(dtype=np.float32)
    output = image + noise
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output

def save(path, filename, img):
    Path(path).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path + filename, img)

if __name__ == "__main__":
    read_dir = ".\\grayscale\\pics\\"
    write_dir = ".\\grayscale\\addnoise\\"

    for file_name in os.listdir(read_dir):
        img0 = cv2.imread(read_dir + file_name, 0)

        img1 = add_Gaussian_noise(img0)
        save(write_dir + "Gaussian_noise\\", file_name, img1)

        for SNR in range(5, 100, 5):
            img2 = add_pepper_salt_noise(img0, SNR)
            save(write_dir + "salt-pepper_noise\\" + str(SNR) + "\\", file_name, img2)

            img3 = add_pepper_salt_noise(img1, SNR)
            save(write_dir + "mixed_noise\\" + str(SNR) + "\\", file_name, img3)