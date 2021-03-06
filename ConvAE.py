from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np


def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):  #构建自编码器模型
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())    
    model.add(Dense(units=filters[3], name='embedding'))   #中间层得到的特征嵌入
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model

if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])   #选择数据集
    parser.add_argument('--n_clusters', default=10, type=int)      #聚类数
    parser.add_argument('--batch_size', default=256, type=int)    #batch大小
    parser.add_argument('--epochs', default=200, type=int)       
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):   #保存路径不存在就创建一个
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps   #从datasets.py中加载数据集
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])    #定义模型
    plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')  #定义模型的优化器与损失函数，mse即均方误差
    from keras.callbacks import CSVLogger     #回调函数，查看函数内部状态
    csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)   #构造一个从输入到嵌入层的编码器模型
    features = feature_model.predict(x)     #获得输入数据对应的输出（即特征嵌入，也就是数据特征的新形式）
    print('feature shape=', features.shape) 

    # use features for clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)   #对特征嵌入进行kmeans，得到聚类中心
    from . import metrics
    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
