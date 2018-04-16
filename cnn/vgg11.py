# -*- coding:utf-8 -*-
from utils import DataType
from utils import gpu_or_cpu
from utils import load_data
from utils import train
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

def train_vgg11():
    num_outputs = 10
    architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            vgg_stack(architecture),
            nn.Flatten(),
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            nn.Dense(4096, activation="relu"),
            nn.Dropout(.5),
            nn.Dense(num_outputs))

    # 初始化
    ctx = gpu_or_cpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    print('initialize weight on', ctx)

    # 获取数据
    batch_size = 64
    resize = 96
    train_data, test_data = load_data(DataType.FASHION_MNIST, batch_size, resize=resize)

    # 训练
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
    train(train_data, test_data, net, loss, trainer, ctx, num_epochs=20)

if __name__ == '__main__':
    train_vgg11()
