# LPG-PCA算法概要

本文基于论文《Two-stage image denoising by principal component analysis with local pixel grouping》写成。

LPG-PCA是一种图像降噪算法，适用于对带有高斯噪声的图片降噪。经实验证明其降噪效果优于传统的小波变换、K-SVD等算法。对于其他分布的噪声降噪效果未知。

LPG-PCA，正如其名所示，该算法有两个核心思想：LPG（局部像素分组）与PCA（主成分分析）。

除了待降噪的图像外，该算法还要求提供一个超参数$\sigma$作为输入，$\sigma$是对图像中高斯噪声标准差的估计，可手动调节。

## LPG（局部像素分组）

该算法将每个像素点看作一个降噪单元，即遍历图像中的所有像素点，依次使用PCA进行降噪，对于任意一个像素点，该算法将其附近的像素点作为它的采样范围，这就是LPG（局部像素分组）。

这样做有两点好处：

1. 我们获得了该点周围的信息
2. 我们在降噪时保留了图像的局部结构

这两点内容将会在接下来部分做进一步讲解。

该算法如下图所示对像素进行分组：

<img width="810" alt="image" src="https://user-images.githubusercontent.com/61410169/122654497-83aa6980-d17e-11eb-905e-327446d4334c.png">

正中间的是我们将要降噪的像素点，我们将以该点为中心，$K\times K$的矩阵所展开的向量称为中心向量；将以该点为中心，$L\times L$的矩阵称为训练窗口，$L$与$K$是超参数，==这里要注意的是，要想以选定的像素点为中心，$K$和$L$应都为奇数==。

训练窗口中每一个$K\times K$的矩阵所展开的向量构成了我们的样本待选集。

我们将样本待选集中的所有向量，按照其与中心向量的均方误差从小到大排序。在这里，对于$m$维的向量$\vec a$与向量$\vec b$，它们的均方误差为：
$$
MSE = \frac1m\Sigma_{i=1}^m(\vec a_i-\vec b_i)^2
$$
其中$\vec a_i, \vec b_i$表示其第$i$个分量。

我们选取样本待选集中均方误差小于$T+2\sigma^2$的向量，如果满足条件的向量个数少于$cK^2$个，则选取均方误差最小的$cK^2$个，这样我们就得到了集合大小不少于$cK^2$的样本集。这里边的$T$与$c$都是我们预先设定的值，$c$一般取$8\sim10$，$T$可手动调节，$\sigma$是超参数，代表我们预估的该图像高斯噪声的标准差。

中心向量代表了我们要降噪的像素点，而样本集中选取的向量都尽可能靠近中心向量，且与中心向量足够相似。因此我们可以从这些向量中获取像素点附近的信息，且在降维时保留局部结构。

## PCA（主成分分析）

接下来我们对该样本集使用PCA降噪，下面简单介绍PCA的数学原理与算法流程，该算法使用的是基于特征值分解的PCA：

PCA是一种常用的降维手段，对于原始数据$X$（若不加说明，以下$X$都是以每一列作为一个样本的矩阵，即$X=[\vec x_1^T,\dots,\vec x_n^T]^T$，其中$\vec x_i$是第$i$个样本的列向量），我们希望找到一组正交基$W$（与$X$同理，每一列是一个基向量），满足最近重构性，即在原空间离得近的样本，在以$W$为基底的新空间中依然离得近。用数学表达就是满足下面的条件：
$$
\min_W-tr(W^TXX^TW)\\
s.t. W^TW=I
$$
$Z=W^TX$即为降维后的结果。

PCA的中文讲解可以参考周志华的《机器学习》和李航的《统计学习方法》，后者从数学的角度详细推导了基于特征值分解与基于奇异值分解的两种PCA方法。

算法流程如下：

1. 对数据做标准化处理，消除量纲
2. 计算样本协方差矩阵$XX^T$
3. 对协方差矩阵$XX^T$做特征值分解，得到一系列特征值与其对应的特征向量
4. 取其中最大的$k$个特征值对应的特征向量构成新基底，即为$W$

## 算法核心思想

PCA虽然是一种降维算法，但它也可以用于降噪，这是因为对于分解后的每一个特征向量，从数学的角度讲，其特征值越大，那么在新空间中，样本在该特征向量表示的维度上越分散（即方差越大）；从信息论的角度讲，特征值越大，熵就越大，以该特征向量作为基向量能够保留的信息就更多，因此方差与信息量在一定程度上可以划等号。

详细的推导和解释可以查阅相关资料，在这里我们只需要知道结论：特征值越大，方差越大，熵越大，那么对应的特征向量就能保留图像更多的信息量。

相对于图片本身的信息而言，图片中噪声的信息量显然要小得多，因此我们只需要去掉那些特征值比较小的特征，即删除图片中占比较少的信息，就可以实现降噪。那么我们应该删除多少信息呢？

在PCA的算法流程中，第4步告诉我们要保留最大的$k$个，剩下的全部删除。但那是对于降维而言，我们要降到$k$维，所以才保留$k$个。但当前我们面对的是降噪情景，那么什么能指导我们降噪呢？

前文提到过，我们输入了一个$\sigma$作为噪声标准差的预估，介绍PCA时也提到了，方差可以衡量信息量的大小，因此$\sigma$可以告诉我们应该删除什么样的信息。

在《Two-stage image denoising by principal component analysis with local pixel grouping》这篇论文中，经过一系列数学推导，作者认为应该删除$\frac{w_i-\sigma^2}{w_i}$趋近于$0$的特征向量，其中$w_i$是第$i$个特征向量对应的特征值。

因此我们对PCA算法流程的第4步稍作修改，将所有特征值取出，记为$W$，然后计算新空间下的样本集$Y=W^TX$，接下来用$\frac{w_i-\sigma^2}{w_i}$乘以$Y$的第$i$行，这样$\frac{w_i-\sigma^2}{w_i}$趋近于$0$的那一行数据就都趋于$0$，这样在变换后的空间中我们就删除了信息量少的维度。最后我们将数据复原$X_{new}=WY$，相当于剔除了样本集中的少量信息，达到了降噪的目的。

很多同学可能不明白为什么$WY$就能够复原信息，这里简单解释一下：$W$是特征向量构成的矩阵，是一个正交阵，而正交阵的定义就是$WW^T=W^TW=I$，即$W^T$就是$W$的逆矩阵，$Y=W^TX$相当于将原样本空间投影到以$W$为基底的空间中，而$X_{new}=WY=(W^T)^{'}Y$其实就是一个逆过程，将新空间中的样本再投影回原空间。只不过我们在新空间中删除了一些维度，投影回原空间后就相当于剔除了原样本的部分信息。

该小节论述的内容也能为前文所说的保留图片的局部结构作出一定解释，因为我们的样本选取的都是离中心向量近的，且与中心向量足够相似的样本点，这就意味着PCA保留下来的信息中，中心向量及其局部结构的信息量占比最大，这也就帮助我们保留了图片的局部结构。

## 分阶段执行

我们已经介绍完了算法的核心思想，但还有一点要说的是，该算法在论文中被分成了两个阶段执行。

第一阶段就像前文说的，我们输入待处理的图像和一个$\sigma$，返回降噪后的图片。但是论文的作者发现，经过这一次降噪后，图像上还保留有肉眼可观察到的噪声，他们通过数学手段分析了噪音残留的原因，然后提出了自适应地调整$\sigma$，即用原$\sigma$自动生成一个$\sigma_{new}$，然后用第一阶段返回的降噪后的图片，和这个新生成的$\sigma_{new}$，再执行一次算法，这就是算法的第二阶段。

$\sigma_{new}$不需要我们手动输入，它是根据如下公式生成的：
$$
\sigma_{new} = c_s\sqrt{MSE(stage1\_image,origin\_image)}
$$
其中$c_s$是一个超参数，实验中发现取$0.35$时效果不错，$stage1\_image$表示第一阶段返回的图像，$origin\_image$是我们输入的待处理图像，它们都是二维矩阵，$MSE(stage1\_image,origin\_image)$是这两个二维矩阵的均方误差。前文介绍过向量的均方误差，在本算法中，对于这两个二维矩阵，他们的均方误差可以看作将这两个二维矩阵展平为一维向量所计算出的均方误差。

## 灰度图像与彩色图像

前文介绍的算法都是基于图像是灰度的，是一个二维矩阵这样的假设，那么对于彩色图像应该如何做呢？

论文的作者给出了两种实现方案：

1. 对于彩色图像的RGB三个通道，分通道降噪，再将结果合并
2. 将LPG中的$K\times K$的矩阵改为$K\times K\times3$的矩阵，$L\times L$的矩阵改为$L\times L\times3$的矩阵，其他步骤相同。

# LPG-PCA算法实现与代码详解

代码用到的库及全局变量：

```python
import os
import numpy as np
from multiprocessing import Pool

global pool
```

前文提到，该算法是以像素点为降噪单元的，而观察算法流程我们也可以发现，像素点之间的降噪过程是彼此无关，互不干扰的，而该算法的计算量偏大，因此可以采用并行计算的手段加快代码执行速度：将图片按行划分，交给多个进程处理。

因为算法分两个阶段，主控程序如下：

```python
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

```

这部分代码注释比较详细，值得注意的就是两个阶段之间$\sigma$的更新，下面介绍每个阶段的程序：

```python
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
    extended_img[L//2:row_size+L//2, L//2:col_size+L//2] = noised_img

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
```

对于该部分代码，值得注意的地方有两点：

第一点就是最后将归一化后的数据复原到0~255的整数空间。因为图像的像素值在0~255这个区间，我们可以通过将dtype参数置为uint8的方式复原。但一定要注意对于大于255和小于0的数据要手动调整，否则会因为数据溢出将其变为错误的数字。

第二点就是图像的增广。我们为什么要对图像增广呢？考虑到算法的流程，我们在LPG阶段要以一个像素点为中心，取$L\times L$的矩阵为训练窗口，而对于边缘的像素点，是没有这样的窗口的，因此我们要将图片的上下左右各增加L/2行，那么增加出来的这些像素点填什么值呢？从理论上讲填充$N(0,\sigma^2)$分布的噪声较为合理，因为PCA阶段我们通过$\sigma$消除了部分信息量，$N(0,\sigma^2)$分布的噪声应该包含在其中。但按照一般做法全都填充为$0$也不无不可，因为对于$\vec 0$来说，它与中心向量的距离过远，被选到作为样本的概率不大，因此也不会影响最后的结果。

接下来就是算法的重头戏，也是前文花费了绝大篇幅介绍的地方：

```python
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
```

相信认真读完前文的同学对于此处的算法应该不难理解，我主要讲解其中重要的一些部分。

第一点，对于训练空间中所有待选样本点的提取，我是通过一层循环实现的，首先计算出待选样本点总数$(L-K+1)^2$，以该训练空间右上角的顶点开始，我们很容易计算出接下来每个顶点的值，因为图片已经增广，所以不会存在溢出的情况。

第二点，对于样本集合的选取，我利用numpy提供的sort函数和argsort函数将（均方误差）从小到大排序，选出了其中的至少$cK^2$个构成样本集合$X$，不熟悉这两个函数的同学可以查阅相关文档。

第三点，数据标准化处理，前文提到过，PCA算法的第1步就是对数据做标准化处理。数据标准化的手段有很多，PCA中常用的有$z\_score$标准化，$min-max$标准化等。因为前面我们曾让图像中的每个像素点都除以255，做了归一化，其实这在图像中就类似于$min-max$标准化，所以在这里我就只做了$z\_score$标准化的一部分，让数据中心化。

第四点，PCA降噪。细心的同学可能发现，这里代码中的公式和前文介绍的PCA公式有些不同，其实是因为这里边的$X$是以每一行作为一个样本的，而前文是以每一列作为一个样本，为了让代码更简洁，我对公式做了一些等价修改，本质上是相同的。

最后一点，数据复原，即返回值。前文中我们说到$X_{new}=WY$可以让数据复原，但这复原的结果是一个矩阵，而我们是要对像素点降维，最后的结果应该是一个像素点，那么我们如何从这个矩阵中得到我们想要的像素点呢？我们想要的像素点其实就是复原后的中心向量的最中间的分量，那么矩阵中的哪个向量是中心向量呢？下面我们详细讲解一下。

首先看下面这句代码，它是整个算法的精髓所在：

```python
    X = eig_vec @ (Y.T[:, 0] * w)
```

原代码中这句话后面还加了一个mean_X，其实就是把数据中心化复原，这里我们并不关心。

上面这句代码等价于下面这句代码：

```python
    X = (eig_vec @ Y.T * w)[:, 0]
```

把[:, 0]，提到里面是为了减少计算量，节省内存，加快运行速度。

我们用这句等价代码讲解，Y.T * w其实就是前文讲到的去除部分信息量的步骤，eig_val @ Y.T * w其实就是前文$X_{new}=WY$的过程，忘记的同学可以去看一眼。这里采用Y.T同样是因为这里的Y是以行作为样本的。[:, 0]就是取出$X_{new}$的第一个向量，为什么要取出第一个？因为它就是复原后中心向量。观察一下MSE的公式就能看出，MSE始终大于等于0，当且仅当两个向量一模一样时，MSE=0，而中心向量是在训练窗口中的，它的MSE为0，这是MSE能取到的最小值，那么它肯定被选入了样本集，而且在第二点中提到过，$X$中的元素是按MSE升序排列的，因此$X$中的第一个向量MSE值一定为0，所以它是一个和中心向量一模一样的向量，我们就可以把它当作中心向量。

拿出复原后中心向量后，它正中间的分量就是降噪后的像素值。因此我们有了如下代码：

```python
    if len(extended_img.shape) == 2:  # 单通道图像返回一个像素值
        return X[K ** 2 // 2]
    else:  # 三通道图像返回RGB三个像素值
        return X[K ** 2 * 3 // 2 - 1:K ** 2 * 3 // 2 + 2]
```

彩色图像要返回RGB三个像素值，其实就是中心向量正中间的三个像素值，至于为什么，很简单，这里就不赘述了，一时间没想懂的可以看看彩色图像的三维矩阵在numpy中是怎么表示的，展平后又是什么样的。

# 代码效果展示

这里给出衡量降噪效果的指标，峰值信噪比（PSNR）的计算公式：
$$
PSNR=20\log_{10}\frac{255}{\sqrt{MSE(origin,denoised)}}
$$
其中orgin是原图，denoised是降噪后的图像，MSE公式如前文，255是像素点最大值，如果你的图像是0~1的灰度图，这里就写1，其他情况同理。

灰度图像：

<img width="858" alt="image" src="https://user-images.githubusercontent.com/61410169/122654553-d71cb780-d17e-11eb-80db-f5899f419b1f.png">

左上角是原图，右上角是加噪后的图片

左下角是第一阶段降噪后的图片

右下角是第二阶段降噪后的图片

两阶段降噪后的峰值信噪比分别为：30.5和31.0

彩色图像：

<img width="845" alt="image" src="https://user-images.githubusercontent.com/61410169/122654514-9c1a8400-d17e-11eb-9c96-eeb77856da90.png">

第一行左面是原图，右面是加噪后的图片

第二行是采用非split方式得到的两阶段的降噪图像

第三行是采用split方式得到的两阶段的降噪图像

这四个图片的峰值信噪比分别为

30.1，31.0

30.6，31.3

# 结语

LPG-PCA是一种有效去除图片高斯噪声的方法，感兴趣的同学也可以在此基础上多多尝试，针对其他类型的噪声，提出创新的想法。

LPG-PCA算法本身也存在一些可以改进或讨论的地方，比如该算法进行了两次迭代，那么如果我让算法多次迭代，直至$\sigma$收敛是否可行呢？算法的某些部分是否还存在进一步的优化空间？数据预处理的手段是否可以更改？等等问题等待大家探索。

本人才疏学浅，若对于相关知识或代码方面有任何疏漏或错误，烦请多多指正。
