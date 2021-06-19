import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from ImageDenoiser import denoise


def psnr(origin, target):
    return 20 * np.log10(255.0 / np.sqrt(np.mean((origin - target) ** 2)))


def test_gray(sigma):
    img = cv.imread('Images/lena.tif')
    plt.subplot(221), plt.imshow(img)
    noise = np.random.normal(0, sigma, (img.shape[0], img.shape[1]))
    noise = np.stack([noise, noise, noise], axis=-1)
    noised_img = noise + img
    noised_img[noised_img > 255] = 255
    noised_img[noised_img < 0] = 0
    noised_img = np.array(noised_img, dtype='uint8')
    plt.subplot(222), plt.imshow(noised_img)
    stage1, stage2 = denoise(noised_img, sigma, ex=True)
    plt.subplot(223), plt.imshow(stage1)
    plt.subplot(224), plt.imshow(stage2)
    print(psnr(img, stage1), psnr(img,stage2))
    plt.show()


def test_color(sigma):
    img = cv.imread('Images/Parrot.tif')
    plt.subplot(321), plt.imshow(img)
    noise = np.random.normal(0, sigma, img.shape)
    noised_img = noise + img
    noised_img[noised_img > 255] = 255
    noised_img[noised_img < 0] = 0
    noised_img = np.array(noised_img, dtype='uint8')
    plt.subplot(322), plt.imshow(noised_img)
    stage1, stage2 = denoise(noised_img, sigma, ex=True)
    plt.subplot(323), plt.imshow(stage1)
    plt.subplot(324), plt.imshow(stage2)
    print(psnr(img, stage1), psnr(img, stage2))
    stage1, stage2 = denoise(noised_img, sigma, ex=True, split=True)
    plt.subplot(325), plt.imshow(stage1)
    plt.subplot(326), plt.imshow(stage2)
    print(psnr(img, stage1), psnr(img, stage2))
    plt.show()


if __name__ == '__main__':
    level = 30
    # test_gray(level)
    test_color(level)
