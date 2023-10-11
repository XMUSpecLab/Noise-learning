"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-13
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig
import numpy as np
import os
from scipy.fftpack import dct, idct
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# define the checkdir
def check_dir(config):
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.test_dir):
        os.mkdir(config.test_dir)


# 定义测试结果保存路径
def test_result_dir(config):
    result_dir = os.path.join(config.test_dir, config.Instrument,
                              config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


# 定义模型保存路径
def save_model_dir(config):
    save_model_dir = os.path.join(config.checkpoint, config.Instrument,
                                  config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    return save_model_dir


# 定义日志路径
def save_log_dir(config):
    save_log_dir = os.path.join(config.logs, config.Instrument,
                                config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    return save_log_dir


def train(config):
    # 指定使用多少个GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    # 使用GPU
    if torch.cuda.is_available():
        # 多GPU训练
        model = nn.DataParallel(model)
        model = model.cuda()
        model.to(device)
    # 定义损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # 定义学习速率调整
    schedual = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)
    # 保存模型的目录
    save_model_path = save_model_dir(config)
    # 保存日志的目录
    save_log = save_log_dir(config)
    # 全局训练步数
    global_step = 0
    # 是否加载预训练模型
    if config.is_pretrain:
        global_step = config.global_step
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        # 加载模型参数
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        # from collections import OrderedDict
        # new_state = OrderedDict()
        # for k, v in state['model'].items():
        # name = 'module.' + k  # add `module.`
        # new_state[name] = v
        # model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
        # model.load_state_dict(new_state)
        model.load_state_dict(state['model'])
        # 恢复优化器状态
        optimizer.load_state_dict(state['optimizer'])
        print('Successfully loaded the pretrained model saved at global step = {}'.format(global_step))
    # 加载数据集
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    # 制作Loader
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)

    train_loader = DataLoader(dataset=TrainSet, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=ValidSet, batch_size=256, pin_memory=True)
    # 仿真光谱生成器
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    # 定义tensorboard
    writer = SummaryWriter(save_log)
    for epoch in range(config.max_epoch):
        for idx, noise in enumerate(train_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_train.generator(spec, spectra_num)
            # 转置
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            # 定义输入输出
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            # 进行预处理, dct变换
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            # reshape 成3维度
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            # 转换为tensor
            input_coef = torch.from_numpy(input_coef)
            output_coef = torch.from_numpy(output_coef)
            # 转换为float类型
            input_coef = input_coef.type(torch.FloatTensor)
            output_coef = output_coef.type(torch.FloatTensor)
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)
            global_step += 1
            model.train()
            optimizer.zero_grad()
            preds = model(input_coef)
            train_loss = criterion(preds, output_coef)
            train_loss.backward()
            optimizer.step()
            # 将训练损失和验证损失写入日志
            writer.add_scalar('train loss', train_loss.item(), global_step=global_step)
            # 每隔xx步打印训练loss
            if idx % config.print_freq == 0:
                print('epoch {}, batch {}, global step  {}, train loss = {}'.format(
                    epoch, idx, global_step, train_loss.item()))

        # 保存模型参数,多GPU一定要用module.state_dict()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'global_step': global_step, 'loss': train_loss.item()}
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        torch.save(state, model_file)
        """
        加载时使用如下代码 
        state = torch.load(save_model_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        """
        # 验证：需使用eval()方法固定BN
        model.eval()
        valid_loss = 0
        for idx_v, noise in enumerate(valid_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            # print(noise.shape)
            clean_spectra = gen_valid.generator(spec, spectra_num)
            # 转置
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            # 定义输入输出
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            # 进行预处理, dct变换
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            # reshape 成3维度
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            # 转换为tensor
            input_coef = torch.from_numpy(input_coef)
            output_coef = torch.from_numpy(output_coef)
            # float tensor
            input_coef = input_coef.type(torch.FloatTensor)
            output_coef = output_coef.type(torch.FloatTensor)
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)
            preds_v = model(input_coef)
            valid_loss += criterion(preds_v, output_coef).item()
        valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar('valid loss', valid_loss, global_step=global_step)
        # 一个epoch完成后调整学习率
        schedual.step()


def test(config):
    print('testing...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 加载模型参数
    state = torch.load(model_file)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 读取测试数据集
    filenames = os.listdir(config.test_data_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 测试数据的绝对路径
            name = config.test_data_root + '/' + file
            # 加载测试数据
            tmp = sio.loadmat(name)
            inpts, inptr = np.array(tmp[config.test_varible[0]]), np.array(tmp[config.test_varible[1]])
            inpts, inptr = inpts.T, inptr.T
            # s-simulated仿真数据, r-real实际数据
            nums, spec = inpts.shape
            numr, _ = inptr.shape
            # DCT 变换
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            for idx in range(numr):
                inptr[idx, :] = dct(np.squeeze(inptr[idx, :]), norm='ortho')
            # 转换为3-D tensor
            inpts, inptr = np.array([inpts]).reshape((nums, 1, spec)), np.array([inptr]).reshape((numr, 1, spec))
            inpts, inptr = torch.from_numpy(inpts), torch.from_numpy(inptr)
            # inptt-total
            inptt = torch.cat([inpts, inptr], dim=0)
            # 划分小batch批量测试
            test_size = 32
            group_total = torch.split(inptt, test_size)
            # 存放测试结果
            predt = []
            # preds, predr = [], []
            for i in range(len(group_total)):
                xt = group_total[i]
                if torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                predt.append(yt)
            predt = torch.cat(predt, dim=0)
            predt = predt.numpy()
            predt = np.squeeze(predt)
            preds, predr = predt[:nums, :], predt[nums:, :]
            for idx in range(nums):
                preds[idx, :] = idct(np.squeeze(preds[idx, :]), norm='ortho')
                predr[idx, :] = idct(np.squeeze(predr[idx, :]), norm='ortho')

            tmp['preds'], tmp['predr'] = preds.T, predr.T
            # 获取存放测试结果目录位置
            test_dir = test_result_dir(config)
            # 新的绝对文件名
            filename = os.path.join(test_dir, "".join(file))
            # 将测试结果保存进测试文件夹
            sio.savemat(filename, tmp)


def predict(config):
    print('predicting...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    save_model_path = save_model_dir(config)
    model_file = os.path.join(save_model_path, str(config.global_step) + '.pt')
    # 加载模型参数
    state = torch.load(model_file)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 定义滑动窗口滤波器

    # def moving_average(data, n):
    #     return np.convolve(data, np.ones(n, )/n, mode='same')

    # 读取需要处理的txt文件
    filenames = os.listdir(config.predict_root)
    i = 0
    fig = plt.figure()

    for file in filenames:
        if os.path.splitext(file)[1] == '.txt':
            # 测试数据的绝对路径
            name = config.predict_root + '/' + file
            new_name = config.predict_root + '/dn_' + file
            # 加载测试数据
            tmp = np.loadtxt(name).astype(np.float)
            wave, x = tmp[:, 0], tmp[:, 1]
            # # 数据预处理 滑动窗口滤波+DCT变换
            # xs = moving_average(x, 3)
            # err = x - xs
            # 作为输入数据
            coe_dct = dct(x, norm='ortho')
            # 更改shape
            inpt = coe_dct.reshape(1, 1, -1)
            # 转换为torch tensor
            inpt = torch.from_numpy(inpt).float()
            # 预测结果
            if torch.cuda.is_available():
                inpt = inpt.cuda()
            yt = model(inpt).detach().cpu().numpy()
            yt = yt.reshape(-1, )
            # idct 变换
            noise = idct(yt, norm='ortho')
            Y = x - noise
            denoised = np.array([wave, Y])
            np.savetxt(new_name, denoised, delimiter='t')
            i = i + 1
            plt.subplot(3, 3, i)
            plt.plot(wave, x)
            plt.plot(wave, Y)
    plt.show()

def batch_predict(config):
    print('batch predicting...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 加载模型参数
    state = torch.load(model_file)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 读取测试数据集
    filenames = os.listdir(config.predict_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 测试数据的绝对路径
            name = config.predict_root + '/' + file
            # 加载测试数据
            tmp = sio.loadmat(name)
            inpts = np.array(tmp['cube'])
            inpts = inpts.T
            nums, spec = inpts.shape
            # DCT 变换
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            # 转换为3-D tensor
            inpts = np.array([inpts]).reshape((nums, 1, spec))
            inpts = torch.from_numpy(inpts)

            # 划分小batch批量测试
            test_size = 32
            group_total = torch.split(inpts, test_size)
            # 存放测试结果
            preds = []
            for i in range(len(group_total)):
                xt = group_total[i]
                if torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                preds.append(yt)
            preds = torch.cat(preds, dim=0)
            preds = preds.numpy()
            preds = np.squeeze(preds)
            for idx in range(nums):
                preds[idx, :] = idct(np.squeeze(preds[idx, :]), norm='ortho')
            tmp['preds'] = preds.T
            # 获取存放测试结果目录位置
            test_dir = test_result_dir(config)
            # 新的绝对文件名
            filename = os.path.join(test_dir, "".join(file))
            # 将测试结果保存进测试文件夹
            sio.savemat(filename, tmp)


def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_testing:
        test(config)
    if config.is_predicting:
        predict(config)
    if config.is_batch_predicting:
        batch_predict(config)


if __name__ == '__main__':
    opt = DefaultConfig()
    main(opt)
