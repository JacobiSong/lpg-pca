import os
import numpy as np
from multiprocessing import Pool

global pool


def _denoise_pixel(extended_img, sigma, x, y, L, K, T, c):
    """
    对一个像素点降噪

    :param extended_img: 增广后的图像
    :param x: 行
    :param y: 列
    :return: 该点降噪后的像素值
    """
    all_num = (L - K + 1) ** 2

    # 提取样本集窗口内的所有待选样本
    samples = []
    for i in range(all_num):
        xx = x + i // (L - K + 1)
        yy = y + i % (L - K + 1)
        samples.append(extended_img[xx:xx + K, yy:yy + K].flatten())
    central = samples[all_num // 2]  # 中心样本
    samples = np.vstack(samples)

    # 选取与中心样本的均方误差符合阈值要求的至少 c*K*K 个样本
    mse = np.mean((samples - central) ** 2, axis=-1)
    samples = samples[np.argsort(mse)]
    mse = np.sort(mse)
    i = min(all_num, c * K * K)
    while i < all_num and mse[i] < T + 2 * sigma ** 2:
        i += 1
    X = samples[:i]

    # 数据中心化
    mean_X = np.mean(X, axis=0)
    X = X - mean_X

    # PCA降噪
    eig_val, eig_vec = np.linalg.eigh(X.T @ X / i)
    Y = X @ eig_vec
    w = np.maximum(0, eig_val - sigma ** 2) / np.maximum(eig_val, 1e-6)
    X = eig_vec @ (Y.T[:, 0] * w) + mean_X

    if len(extended_img.shape) == 2:  # 单通道图像返回一个像素值
        return X[K ** 2 // 2]
    else:  # 三通道图像返回RGB三个像素值
        return X[K ** 2 * 3 // 2 - 1:K ** 2 * 3 // 2 + 2]


def _denoise_row(extended_img, sigma, x, L, K, T, c):
    """
    对图像的一行进行降噪的函数

    :param extended_img: 增广后的图像
    :param x: 行号
    :return: 返回行号与降噪后该行各点的像素值
    """
    pixels = []
    for i in range(extended_img.shape[1] - L + 1):
        pixels.append(_denoise_pixel(extended_img, sigma, x, i, L, K, T, c))
    return x, pixels


def _denoise(noised_img, sigma, L, K, T, c):
    """
    每一阶段的降噪函数
    """
    denoised_img = np.zeros(noised_img.shape)  # 降噪后的图像，待填充
    row_size = noised_img.shape[0]  # 原图像行数
    col_size = noised_img.shape[1]  # 原图像列数

    # 对图像进行增广，将图像的上下左右各扩充L/2维，并用均值为0，标准差为sigma的高斯噪声填充
    if len(noised_img.shape) == 2:
        extended_img_shape = (row_size + L - 1, col_size + L - 1)
    else:
        extended_img_shape = (row_size + L - 1, col_size + L - 1, 3)
    extended_img = np.random.normal(0, sigma, extended_img_shape)
    extended_img[L // 2:row_size + L // 2, L // 2:col_size + L // 2] = noised_img

    # 数据归一化
    extended_img = extended_img / 255.0
    sigma /= 255.0

    def _denoise_row_callback(results: tuple) -> None:
        """
        多进程下完成每一行降噪后的回调函数，生成降噪后的图片

        :param results: _denoise_row()函数返回的结果
        """
        x = results[0]  # 获得行号
        pixels = results[1]  # 获得该行的像素值
        for i in range(col_size):  # 填充该行像素值
            denoised_img[x][i] = pixels[i]

    # 对于图像的每一行创建一个降噪任务
    tasks = [
        pool.apply_async(_denoise_row, (extended_img, sigma, x, L, K, T, c), callback=_denoise_row_callback)
        for x in range(row_size)]

    # 主进程阻塞，直到所有任务完成
    for task in tasks:
        task.wait()

    # 前面做了数据的归一化，因此这里要恢复
    denoised_img = denoised_img * 255
    denoised_img[denoised_img > 255] = 255
    denoised_img[denoised_img < 0] = 0
    return np.array(denoised_img, dtype='uint8')


def denoise(noised_img, sigma=10, L=11, K=3, T=0, c=8, cs=0.35, split=False, ex=False):
    """
    带有高斯噪音的图像的降噪函数

    :param noised_img: 待降噪的图像
    :param sigma: 噪音等级，即噪音的标准差
    :param L: 训练窗口大小
    :param K: 单个样本的窗口大小
    :param T: 采样相似度阈值
    :param c: 用于控制样本数量最小值，一般取8~10
    :param cs: 小于1，用于自适应调整第二阶段噪音等级，一般取0.35
    :param split 该参数开启后，对于彩色图像，RGB三个通道将分开降噪，结果合并后返回
    :param ex: 该参数开启后会分别输出两个阶段的图像
    :return: 降噪后的图像
    """

    # 计算图像通道数
    layers = len(noised_img.shape)
    # 判断图像是否为彩色
    is_colorful = False
    if len(noised_img.shape) == 3:
        for i in range(1, 3):
            if not np.all(noised_img[:, :, i] == noised_img[:, :, i - 1]):
                is_colorful = True
                break
    elif layers != 2:
        return None

    # 创建进程池
    global pool
    pool = Pool(os.cpu_count() - 1)

    if is_colorful and split:  # 对于开启split参数的彩色图像，分通道降噪
        # 第一阶段
        stage1 = []
        for i in range(3):  # 3个通道分别降噪
            stage1.append(_denoise(noised_img[:, :, i], sigma, L, K, T, c))
        stage1 = np.stack(stage1, axis=-1)  # 结果合并
        # 更新sigma
        sigma = cs * np.sqrt(sigma ** 2 - np.mean((stage1 - noised_img) ** 2))
        # 第二阶段
        stage2 = []
        for i in range(3):
            stage2.append(_denoise(stage1[:, :, i], sigma, L, K, T, c))
        stage2 = np.stack(stage2, axis=-1)
    elif is_colorful or layers == 2:  # 对于关闭split参数的彩色图像或单通道图像直接进行降噪
        stage1 = _denoise(noised_img, sigma, L, K, T, c)
        sigma = cs * np.sqrt(sigma ** 2 - np.mean((stage1 - noised_img) ** 2))
        stage2 = _denoise(stage1, sigma, L, K, T, c)
    else:  # 对于多通道的灰度图像，我们取其中一个通道进行降噪，并将结果复原为多通道
        stage1 = _denoise(noised_img[:, :, 0], sigma, L, K, T, c)  # 取其中一个通道降噪
        stage1 = np.stack([stage1, stage1, stage1], axis=-1)  # 结果拷贝3份后直接合并
        sigma = cs * np.sqrt(sigma ** 2 - np.mean((stage1 - noised_img) ** 2))
        stage2 = _denoise(stage1[:, :, 0], sigma, L, K, T, c)
        stage2 = np.stack([stage2, stage2, stage2], axis=-1)

    # 释放线程池
    pool.close()
    pool.join()

    if ex:
        return stage1, stage2
    else:
        return stage2
