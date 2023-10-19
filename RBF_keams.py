from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class RBF(object):
    def __init__(self, k_rbf, train_x_rbf, train_y_rbf, method_rbf):

        self.k = k_rbf  # 神经元的数量
        self.train_x_rbf = train_x_rbf  # 训练数据的输入
        self.train_y_rbf = train_y_rbf  # 训练数据输入的真实值
        self.method_rbf = method_rbf  # 计算高斯核函数所用的方差所用到方法
        self.centers_rbf = []  # 神经元中心
        self.max_centers_distance_rbf = None  # 神经元之间的最大距离
        self.variances_rbf = None  # 计算高斯核函数所用的方差
        self.weights_rbf = None  # 输出权重
        self.bias_rbf = None  # 偏置

    def k_means(self):
        k_means = KMeans(n_clusters=self.k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        k_means.fit(self.train_x_rbf)  # 使用训练数据进行聚类
        self.centers_rbf = k_means.cluster_centers_  # 得到聚类中心
        distances_km = pairwise_distances(self.centers_rbf, self.centers_rbf)  # 计算聚类中心之间距离
        np.fill_diagonal(distances_km, 0)  # 将对角线元素设为0
        self.max_centers_distance_rbf = np.max(distances_km)  # 得到聚类中心之间距离的最大值

    def weights_bias_calculate(self):
        variances_broadcast_rbf = None
        train_x_centers_distance = pairwise_distances(self.train_x_rbf, self.centers_rbf)  # 计算输入数据到聚类中心之间距离

        if self.method_rbf == 'max_centers_distance':  # 判断计算高斯核函数所用的方差所用到方法
            self.variances_rbf = self.max_centers_distance_rbf
            variances_broadcast_rbf = np.full_like(train_x_centers_distance, self.variances_rbf)  # 将variances_rbf广播成train_x_centers_distance的形状

        elif self.method_rbf == 'max_centers_distance_divided_by_k':
            self.variances_rbf = self.max_centers_distance_rbf/np.sqrt(2*self.k)
            variances_broadcast_rbf = np.full_like(train_x_centers_distance, self.variances_rbf)
        hidden_layer_output_rbf = np.exp(-(train_x_centers_distance / variances_broadcast_rbf) ** 2)  # 计算隐含层输出
        ones_column = np.ones((hidden_layer_output_rbf.shape[0], 1))  # 创建一个形状为(centers_centers_distance.shape[0], 1) 的全为1的列
        hidden_layer_output_extend_rbf = np.hstack((hidden_layer_output_rbf, ones_column))  # 水平拼接隐含层输出和全为1的列，变为为广义rbf
        hidden_layer_output_extend_inverse_rbf = np.linalg.pinv(hidden_layer_output_extend_rbf)  # 求伪逆矩阵
        weights_bias_extend_rbf = np.dot(hidden_layer_output_extend_inverse_rbf, self.train_y_rbf.T)  # 计算权重与偏置项
        self.weights_rbf = np.squeeze(weights_bias_extend_rbf)[0:self.k]
        self.bias_rbf = np.squeeze(weights_bias_extend_rbf)[self.k]

    def return_data(self):
        return self


def rbf_predict(x_rp, weights_rp, bias_rp, centers_rp, variances_rp):
    if np.ndim(x_rp) == 1:
        x_rp = x_rp[np.newaxis, :]
    distance_rp = pairwise_distances(x_rp, centers_rp)
    variances_extend_rp = np.full_like(distance_rp, variances_rp)
    hidden_layer_output_rp = np.exp(-(distance_rp / variances_extend_rp) ** 2)
    y_rp = np.dot(hidden_layer_output_rp, weights_rp.T) + bias_rp
    return y_rp


if __name__ == '__main__':
    num = 0
    ub = 1.0  # 上边界
    lb = 0.0  # 下边界
    n = 100  # 数据的数量
    d = 1  # 数据的维度
    k = [3, 6, 8, 10, 15]  # 聚类中心的个数
    method = ['max_centers_distance', 'max_centers_distance_divided_by_k']
    noise = np.random.uniform(low=-0.1, high=0.1, size=(n, d))
    sample = lhs(d, samples=n)  # LHS采样
    x = sample * (ub - lb) + lb
    y_noise = (0.5 + (0.4 * np.cos((x * np.pi * 2.5)))) + noise
    y_actual = (0.5 + (0.4 * np.cos((x * np.pi * 2.5))))

    for i_k in k:
        for j_method in method:
            rbf = RBF(i_k, x, y_noise.T, j_method) 
            rbf.k_means()
            rbf.weights_bias_calculate()
            data = rbf.return_data()

            weights = data.weights_rbf
            bias = data.bias_rbf
            centers = data.centers_rbf
            variances = data.variances_rbf
            x_predict = lhs(d, samples=n) * (ub - lb) + lb
            y_predict = rbf_predict(x_predict, weights, bias, centers, variances)

            plt.figure(num=num)
            title_text = 'k numbers:{} \n calculate variances method:{}'.format(i_k, j_method)
            plt.scatter(x, y_actual, c='#ED5C27', label='y_actual')  # 绘制图像
            plt.scatter(x, y_noise, c='#40E0D0', label='y_noise')  # 绘制图像
            plt.scatter(x_predict, y_predict, c='#C0FF3E', label='y_predict')  # 绘制图像
            plt.legend(loc='lower right')   # 添加图例
            plt.title(title_text)  # 添加标题
            num += 1
    plt.show()  # 图像展示
