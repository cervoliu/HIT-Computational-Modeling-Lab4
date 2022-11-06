github repository : https://github.com/cervoliu/HIT-Computational-Modeling-Lab4

目录解释见本文档末尾的 project tree

--- 

# 图像去噪
**由于计算量巨大，运行时间较长！**

仅实现grayscale图像

## verification
添加噪声:
```
python make_noise.py
```

去除噪声：

salt-pepper_filter.py 使用 cv2.medianBlur()

my_salt-pepper_filter.py 使用 手写中值滤波（比前者慢许多，慎用！）

```
python Gaussian_filter.py
python salt-pepper_filter.py or python my_salt-pepper_filter.py
python mixed_filter.py
```

## idea
- 仅含有椒盐噪声：中值滤波
- 仅含有高斯噪声：高斯滤波（高斯模糊）
- 混合噪声（先添加高斯噪声，再添加椒盐噪声）：先中值滤波，再高斯滤波

## PSNR与SSIM的计算

PSNR: Peak Signal-to-Noise Ratio

MSE: Mean Square Error

$\mathrm{MAX_f}$ 表示图像点颜色的最大数值，对于8-bit图像为255

$$\mathrm{PSNR} = 20 \log_{10}(\dfrac{\mathrm{MAX_f}}{\sqrt{\mathrm{MSE}}})$$

$$ \mathrm{MSE} = \dfrac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1}  [f(i,j)-g(i,j)]^2$$

SSIM: Structural Similarity

$$ \mathrm{SSIM}(x,y)=\dfrac{(2\mu_{x}\mu_{y}+c_1)(2\sigma_{xy}+c_2)}{(\mu_{x}^2+\mu_{y}^2+c_1)(\sigma_{x}^2+\sigma_{y}^2+c_2)}$$

with
- $\mu_{x}$ the pixel sample mean of $x$
- $\mu_{y}$ the pixel sample mean of $y$
- $\sigma_{x}^2$ the variance of $x$
- $\sigma_{y}^2$ the variance of $y$
- $\sigma_{xy}$ the covariance of $x$ and $y$
- $c_1=(k_1L)^2, c_2=(k_2L)^2$ two variables to stabilize the division with weak denominator
- $L$ the dynamic range of the pixel-values (for 8-bit, it's 255)
- $k_1=0.01$ and $k_2=0.03$ by default

---

# 高斯背景建模

## verification
```
python background_modeling.py
```

---

project tree

```
lab4
├─ background_modeling.py 背景建模源代码
├─ color 灰度图像对应的彩色图像
│  ├─ animal_c.bmp
│  ├─ landscape1_c.bmp
│  ├─ landscape2_c.bmp
│  ├─ lena_c.bmp
│  ├─ pigeon_c.bmp
│  └─ yukino_c.bmp
├─ Gaussian_filter.py 高斯去噪源代码
├─ grayscale 灰度图像实验结果
│  ├─ addnoise 图像添加噪声后的结果
│  │  ├─ Gaussian_noise 添加高斯噪声
│  │  │  ├─ ...
│  │  │  └─ yukino.bmp
│  │  ├─ mixed_noise 添加混合噪声（先高斯，后椒盐）
│  │  │  ├─ ...
│  │  │     ├─ ...
│  │  │     └─ yukino.bmp
│  │  └─ salt-pepper_noise 添加椒盐噪声
│  │     ├─ ...
│  │     │  ├─ ...
│  │     │  └─ yukino.bmp
│  ├─ filter 加噪图像去除噪声（滤波）后的结果（使用cv2库函数滤波）
│  │  ├─ Gaussian_filter 对 Gaussian_noise 滤波
│  │  │  ├─ ...
│  │  │  ├─ yukino.bmp
│  │  │  └─ result.txt
│  │  ├─ mixed_filter 对 mixed_noise 滤波
│  │  │  ├─ ...
│  │  │  │  ├─ ...
│  │  │  │  ├─ yukino.bmp
│  │  │  │  └─ result.txt
│  │  └─ salt-pepper_filter 对 salt-pepper_noise 滤波
│  │  │  ├─ ...
│  │  │  │  ├─ ...
│  │  │  │  ├─ yukino.bmp
│  │  │  │  └─ result.txt
│  ├─ myfilter 加噪图像去除椒盐噪声（中值滤波）后的结果（手写中值滤波；因效率问题，仅对 lena.bmp 测试）
│  │  └─ salt-pepper_filter
│  │     ├─ ...
│  │        ├─ lena.bmp
│  │        └─ result.txt
│  └─ pics 原始灰度图像
│     ├─ animal.bmp
│     ├─ landscape1.bmp
│     ├─ landscape2.bmp
│     ├─ lena.bmp
│     ├─ pigeon.bmp
│     └─ yukino.bmp
├─ make_noise.py  添加噪声
├─ mixed_filter.py 混合去噪
├─ my_salt-pepper_filter.py 椒盐去噪（手写中值滤波）
├─ filter_new.py  自适应窗口大小中值滤波
├─ README.md 本文件
├─ salt-pepper_filter.py 椒盐去噪（使用cv2库函数）
└─ video
   └─ test.avi  背景建模测试视频

```