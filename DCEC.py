from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import metrics
from ConvAE import CAE


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)  #将值传给从父类layer继承来的属性中
        self.n_clusters = n_clusters                     #子类中新增的属性
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)    #设置为2维

    def build(self, input_shape):
        assert len(input_shape) == 2      #由于维度是2，所以shape的长度也是2，assert判断表达式，为false时触发异常
        input_dim = input_shape[1]        #shape第一维是样本数，第二维是样本的维度，这里为了得到样本的维度
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim)) 
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters') #shape第一维是聚类数，第二维是维度
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)  #如果有传入初始权重就设置为初始权重weights
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        import keras.backend as K
        将聚类中心作为可训练的权值，通过t分布将嵌入点映射为软标签
        K.expand_dim在inputs中增加一维
        K.sum计算张量指定轴的和
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha)) 
        q **= (self.alpha + 1.0) / 2.0  #c **= a 等效于 c = c ** a
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q     #qij代表了zi属于类别j的可能性

    def compute_output_shape(self, input_shape):     #修改了输入的shape，在这里指定shape变化的方法
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):     #获取当前层的参数信息
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0):

        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = CAE(input_shape, filters)     #调用自编码器
        hidden = self.cae.get_layer(name='embedding').output    #将嵌入层得到的嵌入特征保留
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)   #由cae的输入到嵌入层构建encoder模型

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)   #将嵌入层的输出作为聚类层的输入
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output])      #定义DCEC的模型（从cae的输入开始，最后输出有cae的输出和聚类层的输出）

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam', save_dir='results/temp'): #预训练：对自编码器的参数
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(args.save_dir + '/pretrain_log.csv')  #将epoch结果流式传输到csv文件的回调

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])  #执行cae模型
        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)   #读取权重

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)   #使用encoder模型得到数据x的特征嵌入（即提取数据的特征）

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):    #得到目标分布P
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):  #重写编译函数（对网络学习过程进行配置）
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):  #重写运行函数fit

        print('Update interval', update_interval)
        save_interval = x.shape[0] / batch_size * 5
        print('Save interval', save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:   #如果没有进行预训练且参数为空则预训练
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:      #参数不为空则加载参数
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])  #使用kmeans得到初始的聚类中心传入聚类层

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:  #每T次迭代使用所有嵌入点更新目标分布
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance使用新分布更新聚类性能
                self.y_pred = q.argmax(1) 
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion  #检查是否收敛（标签分配变化小于阈值）
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model   按保存间隔保存中间模型
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print('saving model to:', save_dir + '/dcec_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/dcec_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save_weights(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'mnist-test'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/temp')
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')
    elif args.dataset == 'mnist-test':
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]

    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=args.n_clusters)
    plot_model(dcec.model, to_file=args.save_dir + '/dcec_model.png', show_shapes=True)
    dcec.model.summary()

    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)  #kld即kl散度
    dcec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights)
    y_pred = dcec.y_pred
    print('acc = %.4f, nmi = %.4f, ari = %.4f' % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred)))
