"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-13
"""

import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt


class spectra_generator:
    def __init__(self):
        self = self
        # self.L = L
        # self.batch_size = batch_size

    # 定义pesudo-voigt函数
    def pVoigt(self, pos, fw, L):
        x = np.arange(0.1, L, 1)
        # Gaussian part
        sigma = fw / (2 * np.sqrt(2 * np.log(2)))
        g = 1 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-(x - pos) ** 2 / (2 * sigma ** 2))
        # Lorentzian part
        l = (1 / math.pi) * (fw / 2) / ((x - pos) ** 2 + (fw / 2) ** 2)
        # final pesudo-Voigt data
        alpha = 0.6785
        pv = (1 - alpha) * g + alpha * l
        # Normalization
        vmin = np.min(pv)
        vmax = np.max(pv)
        pv = (pv - vmin) / (vmax - vmin)
        return pv

    def generator(self, L=None, batch_size=None):
        spectra = np.zeros((L, batch_size))
        for idx in range(batch_size):
            # generate peaks
            for p in range(rd.randint(1, 8)):
                pos = rd.randint(5, L - 5)
                fw = rd.randint(5, 100)
                spectra[:, idx] += self.pVoigt(pos, fw, L) * rd.randint(1, 1000)
        return spectra

    # 定义生成双峰的光谱数据
    def two_peak(self, L=None, X=None, deltas=None, fws=None):
        # spectra = np.zeros((L, len(deltas)*len(fws)))
        spectra = []
        for x in X:
            for delta in deltas:
                for fw in fws:
                    # for training dataset
                    # spectrum = self.pVoigt(x, fw, L) + self.pVoigt(x + delta, fw, L)
                    # spectrum = spectrum * rd.randint(5, 20)
                    # for testing dataset
                    spectrum = self.pVoigt(x, fw, L) + self.pVoigt(x + delta, fw, L)
                    spectrum = spectrum * 10
                    spectra.append(spectrum)
        data = np.array(spectra)
        return data

    # 定义生成强度序列数据集
    def peak_series(self, L=None, X=None, fws=None, intens=None):
        spectra = []
        for fw in fws:
            for inten in intens:
                spectrum = self.pVoigt(X, fw, L) * inten
                spectra.append(spectrum)
        data = np.array(spectra)
        return data


if __name__ == '__main__':
    gen = spectra_generator()
    # training set parameters
    # X = np.linspace(200, 1400, 50, dtype=int)
    # deltas = range(0, 50, 1)
    # fws = np.linspace(5, 200, 50, dtype=int)

    # testing set parameters
    # X = [750]
    # deltas = range(0, 100, 1)
    # fws = np.linspace(5, 200, 50, dtype=int)
    # gt = gen.two_peak(1600, X, deltas, fws)
    # print(gt.shape)
    X = [750]
    fws = np.ones(1)*100
    deltas = range(150, 180, 1)
    gt = gen.two_peak(1600, X, deltas, fws)
    print(gt.shape)

    '''
    # 创建条纹结构
    # line step data
    X = [750]
    fws = np.ones(110) * 50
    # intens = np.concatenate([np.ones(3) * 10, np.ones(2) * 12])
    # intens = np.tile(intens, 6)
    series = range(1, 11)
    downstair, upstair = 10, 15
    intens = []
    for idx in series:
        intens.append(np.tile(downstair, int(idx)))
        intens.append(np.tile(upstair, int(idx)))
        # if np.mod(idx, 2) == 1:
        #     intens.append(np.tile(downstair, int(idx)))
        # else:
        #     intens.append(np.tile(upstair, int(idx)))
    print(intens)
    intens = np.concatenate(intens)
    print(intens.shape)
    gt = gen.peak_series(1600, X, fws, intens)
    print(gt.shape)
    '''
    # sigmoid step data
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # X = [750]
    # # fws = [10, 20, 30, 40, 50]
    # fws = np.ones(30)*50
    # num = np.linspace(-10, 10, 30)
    # intens = sigmoid(num)*10+10
    # plt.figure(1)
    # plt.plot(intens)
    # plt.show()
    # gt = gen.peak_series(1600, X, fws, intens)
    # 加载噪音数据
    noise = np.load('noise.npy')
    noise = noise.T
    print(noise.shape)
    x1 = gt.shape[0]
    x2 = noise.shape[0]
    repts = int(np.ceil(x1 / x2))
    print(repts)
    big_noise = np.tile(noise, (repts, 1))
    print(big_noise.shape)
    # shuffle the noise
    indices = np.random.permutation(big_noise.shape[0])
    big_noise = big_noise[indices]
    print(big_noise.shape)
    noisy = gt + big_noise[:x1]
    dataset = {'gt': gt, 'noisy': noisy}
    print(noisy.shape)
    print(gt.shape)
    np.save('dualpkset(fw100).npy', dataset)
    print('File saved success!')
    plt.figure(1)
    for idx in range(16):
        plt.subplot(4, 4, idx + 1)
        ind = rd.randint(1, gt.shape[0])
        plt.plot(noisy[idx, :], label='Noisy')
        plt.plot(gt[idx, :], label='Clean')
        plt.legend()
    plt.show()
