github repository : https://github.com/cervoliu/HIT-Computational-Modeling-Lab4

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
├─ background_modeling.py
├─ color
│  ├─ animal_c.bmp
│  ├─ landscape1_c.bmp
│  ├─ landscape2_c.bmp
│  ├─ lena_c.bmp
│  ├─ pigeon_c.bmp
│  └─ yukino_c.bmp
├─ filter.py
├─ Gaussian_filter.py
├─ grayscale
│  ├─ addnoise
│  │  ├─ Gaussian_noise
│  │  │  ├─ ...
│  │  │  └─ yukino.bmp
│  │  ├─ mixed_noise
│  │  │  ├─ ...
│  │  │     ├─ ...
│  │  │     └─ yukino.bmp
│  │  └─ salt-pepper_noise
│  │     ├─ ...
│  │     │  ├─ ...
│  │     │  └─ yukino.bmp
│  ├─ filter
│  │  ├─ Gaussian_filter
│  │  │  ├─ ...
│  │  │  ├─ yukino.bmp
│  │  │  └─ result.txt
│  │  ├─ mixed_filter
│  │  │  ├─ ...
│  │  │  │  ├─ ...
│  │  │  │  ├─ yukino.bmp
│  │  │  │  └─ result.txt
│  │  └─ salt-pepper_filter
│  │  │  ├─ ...
│  │  │  │  ├─ ...
│  │  │  │  ├─ yukino.bmp
│  │  │  │  └─ result.txt
│  ├─ myfilter
│  │  └─ salt-pepper_filter
│  │     ├─ ...
│  │        ├─ lena.bmp
│  │        └─ result.txt
│  └─ pics
│     ├─ animal.bmp
│     ├─ landscape1.bmp
│     ├─ landscape2.bmp
│     ├─ lena.bmp
│     ├─ pigeon.bmp
│     └─ yukino.bmp
├─ make_noise.py
├─ measure.py
├─ mixed_filter.py
├─ myfilter.py
├─ my_salt-pepper_filter.py
├─ README.md
├─ salt-pepper_filter.py
└─ video
   └─ test.avi

```