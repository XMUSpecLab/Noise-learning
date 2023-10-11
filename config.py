"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-16
"""

import warnings
import torch


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model_name = 'UNet_CA'
    # 定义保存模型目录
    checkpoint = './Checkpoints'
    # 定义日志保存目录
    logs = './Logs'
    # 定义测试结果保存路径
    test_dir = './Result'
    # 定义数据来源
    # Instrument = 'Hariba'
    Instrument = 'Nanophoton'
    # 测试模型的路径，需更改
    # Hariba
    # test_model_dir = r'H:\Projects\Instrumental noise modeling\code\VA4\Checkpoints\Hariba\UNet_CA\batch_64'
    # nanophoton
    test_model_dir = r'H:\Projects\Instrumental noise modeling\code\VA4\Checkpoints\Nanophoton\UNet_CA\batch_64'
    # 训练集存放路径
    train_data_root = r'H:\PAPER\paper writing\Noise learning\Simulate datasets'
    # 测试集存放路径
    test_data_root = r'H:\PAPER\paper writing\Noise learning'
    # 预测数据集存放路径
    # predict_root = r'H:\PAPER\paper writing\Noise learning\数据\TERS\20220721 PIC'
    predict_root = r'H:\PAPER\paper writing\Noise learning\修改Revision\光谱空间分辨率\gaoyun\nanophoton\test'
    batch_size = 64  # batch size
    print_freq = 50  # print info every N batch
    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    # 定义训练/测试状态/预测状态
    is_training = False
    is_pretrain = False
    is_testing = False
    is_predicting = False
    is_batch_predicting = True

    # 是否使用gpu
    use_gpu = True
    # 加载哪一步的模型
    # horiba
    # global_step = 562800
    # Nanophoton
    global_step = 850000
    # 定义验证数据集的比例
    valid_ratio = 20
    # 定义输入数据名称
    test_varible = ['lcube', 'cube']

    def _parse(self, kwargs, opt=None):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():  # 字典items方法返回可遍历的(键, 值) 元组数组。
            if not hasattr(self, k):  # 若没有k属性打印warning
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)  # 设置k属性的值

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
        '''
        torch.device代表将torch.Tensor分配到的设备的对象。
        torch.device包含一个设备类型（'cpu'或'cuda'设备类型）
        和可选的设备的序号。如果设备序号不存在，则为当前设备; 
        例如，torch.Tensor用设备构建'cuda'的结果等同于'cuda:X',
        其中X是torch.cuda.current_device()的结果。
        torch.Tensor的设备可以通过Tensor.device访问属性。
        构造torch.device可以通过字符串/字符串和设备编号。
        '''

        print('user config:')
        for k, v in self.__class__.__dict__.items():  # 实例对应的类的属性
            if not k.startswith('__'):  # 如果不是以'__'开头
                print(k, getattr(self, k))  # 打印值
