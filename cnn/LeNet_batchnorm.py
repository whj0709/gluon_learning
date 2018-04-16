# -*- coding:utf-8 -*-
from utils import DataType
from utils import gpu_or_cpu
from utils import load_data
from utils import train
from mxnet import gluon
from mxnet.gluon import nn

def train_lenet_batchnorm():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=20, kernel_size=5),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=50, kernel_size=3),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(128, activation="relu"),
            nn.Dense(10)
        )

    # 初始化
    ctx = gpu_or_cpu()
    net.initialize(ctx=ctx)
    print('initialize weight on', ctx)

    # 获取数据
    batch_size = 256
    train_data, test_data = load_data(DataType.FASHION_MNIST, batch_size)

    # 训练
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)

if __name__ == '__main__':
    train_lenet_batchnorm()
