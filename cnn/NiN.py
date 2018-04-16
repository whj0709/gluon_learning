# -*- coding:utf-8 -*-
from utils import DataType
from utils import gpu_or_cpu
from utils import load_data
from utils import train
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn

def mlpconv(channels, kernel_size, padding, strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1, padding=0, strides=1, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1, padding=0, strides=1, activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out

def train_nin():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            mlpconv(96, 11, 0, strides=4),
            mlpconv(256, 5, 2),
            mlpconv(384, 3, 1),
            nn.Dropout(.5),
            mlpconv(10, 3, 1, max_pooling=False),
            nn.GlobalAvgPool2D(),
            nn.Flatten()
        )

    # 初始化
    ctx = gpu_or_cpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    print('initialize weight on', ctx)

    # 获取数据
    batch_size = 64
    resize = 224
    train_data, test_data = load_data(DataType.FASHION_MNIST, batch_size, resize=resize)

    # 训练
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    train(train_data, test_data, net, loss, trainer, ctx, num_epochs=20)

if __name__ == '__main__':
    train_nin()
