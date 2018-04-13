# -*- coding:utf-8 -*-
from enum import Enum
from time import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import image
from mxnet import autograd

class DataType(Enum):
    '''
    数据类型
    '''
    FASHION_MNIST = 0

class DataLoader(object):
    '''
    内存数据加载类
    '''
    def __init__(self, dataset, batch_size, shuffle, transform=None):
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__transform = transform

    def __iter__(self):
        data = self.__dataset[:]
        X = data[0]
        y = nd.array(data[1])
        n = X.shape[0]
        if self.__shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.__batch_size):
            start_index = i*self.__batch_size
            end_index = (i+1)*self.__batch_size
            if self.__transform is not None:
                yield self.__transform(X[start_index:end_index], y[start_index:end_index])
            else:
                yield (X[start_index:end_index], y[start_index:end_index])

    def __len__(self):
        return len(self.__dataset)//self.__batch_size

def gpu_or_cpu():
    '''
    如果有gpu则返回gpu，否则返回cpu
    '''
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def load_data(data_type, batch_size, resize=None):
    '''
    加载数据
    '''
    train_data = None
    test_data = None

    if data_type is DataType.FASHION_MNIST:
        train_data, test_data = __load_fashion_mnist(batch_size, resize=resize)

    return (train_data, test_data)

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    '''
    训练
    '''
    print("start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    for epoch in range(num_epochs):
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()

        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = __get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses.extend([loss(yhat, y) for yhat, y in zip(outputs, label)])
            for l in losses:
                l.backward()

            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar() for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)

            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i+1) % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (n, train_loss/n, train_acc/m))

        test_acc = __evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start))

def __load_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    '''
    加载fashion-mnist数据
    '''
    def transform(data, label):
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False)
    train_data = DataLoader(mnist_train, batch_size, shuffle=True, transform=transform)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False, transform=transform)

    return (train_data, test_data)

def __get_batch(batch, ctx):
    '''
    在ctx上返回data和label
    '''
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch

    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def __evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    '''
    计算准确率
    '''
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()

    for batch in data_iterator:
        data, label, _ = __get_batch(batch, ctx)
        for X, y in zip(data, label):
            y = y.astype('float32')
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()

    return acc.asscalar() / n
