# -*- coding:utf-8 -*-
from utils import DataType
from utils import gpu_or_cpu
from utils import load_data
from utils import train
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn

def train_alexnet():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            # 第一阶段
            nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第二阶段
            nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第三阶段
            nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 第四阶段
            nn.Flatten(),
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            # 第五阶段
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            # 第六阶段
            nn.Dense(10)
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
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    train(train_data, test_data, net, loss, trainer, ctx, num_epochs=20)

if __name__ == '__main__':
    train_alexnet()
