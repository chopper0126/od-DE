import itertools
import time

import numpy as np

import datetime
import os

from matplotlib import pyplot as plt
from numpy import cos, pi, sin
import matplotlib as mpl
import pandas as pd
import datetime
import os
import argparse

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class ARDPSO():
    '''
    ARDPSO算法：
    特点：   1.
            2.
            3.
    输入：   fitness:适应度函数
            constraints:约束条件
            lowwer:下界
            upper:上界
            pop_size:种群大小
            dim:维度
            速度更新公式:变异方式
            epochs:迭代次数
    输出：   best:最优个体

    本程序在原JADE算法上增加约束处理策略，使其能使用于约束优化问题
    '''
    def __init__(self, fitness, constraints, lowwer, upper, pop_size, dim, key, epochs,sample_size,good_p,stdev_data):
        self.fitness = fitness  # 适应度函数
        self.constraints = constraints  # 约束条件
        self.lowbound = lowwer  # 下界
        self.upbound = upper  # 上界
        self.pop_size = pop_size  # 种群大小
        self.dim = dim  # 维度
        self.sample_size = sample_size  # 维度
        self.population = np.zeros((self.pop_size, self.dim))  # 种群位置
        self.v = np.zeros((self.pop_size, self.dim))   # 种群速度
        self.pbest = np.zeros((self.pop_size, self.dim))     # 针对每个粒子来说的历史最优位置
        self.pbest_fit = 1e15
        self.gbest = self.population[0] # 最优个体
        self.gbest_fit = 1e15
        self.fit = np.random.rand(self.pop_size) # 适应度
        self.conv=np.random.rand(self.pop_size) # 约束
        self.key = key # 速度更新公式
        # 正态分布参数
        self.good_p = good_p # good个体
        self.sample_size = sample_size  # 维度
        self.stdev_data = stdev_data  # 标准差

        # PSO的参数
        self.yuzhi = 1
        # self.w = 2  # 惯性因子，一般取1
        self.α = 0.9
        self.c1 = 2  # 学习因子，一般取2
        self.c2 = 2  #
        self.max_vel = 0.5 # 限制粒子的最大速度为0.5
        self.r1 = None  # 为两个（0,1）之间的随机数
        self.r2 = None  # 为两个（0,1）之间的随机数
        #***************************************
        self.Epochs = epochs  # 迭代次数
        self.NFE = 0  # 记录函数调用次数
        self.max_NFE = 10000*self.dim  # 最大函数调用次数
    def init_pop(self):
        self.population = np.random.uniform(self.lowbound, self.upbound, size=(self.pop_size, self.dim))  # 种群
        self.population[:self.sample_size] = self.generate_good_population(self.good_p,self.sample_size,self.dim,self.stdev_data)
        self.v = np.random.uniform(-self.max_vel,self.max_vel, size=(self.pop_size, self.dim))  # 初始化种群速度
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])  # 约束初始化
        self.NFE += self.pop_size  # 记录函数调用次数
        self.pbest = self.population # 每个粒子历史最优个体初始化位置
        self.gbest = self.population[np.argmin(self.fit)]  # 当代最优个体初始化位置
        self.gbest_fit = np.min(self.fit)                  # 当代最优个体的适应度

    def generate_good_population(self,good_p, samlpe_size, dim,stdev_data):
        good_population = np.zeros((samlpe_size, dim))
        #
        for i in range(len(good_p)):
            mean_data = good_p[i]
            m_fenbu = np.random.normal(mean_data, stdev_data, 20)
            good_population[:, [i]] = np.array([m_fenbu]).T
        return good_population

    def run(self):
        fitness = []
        fitness_mean = []
        count = 0
        pre = 0
        self.init_pop()  # 初始化种群
        # 开始迭代
        for j in range(self.Epochs):
            # RDPSO自适应权重w
            w = self.cal_w(self.Epochs, j + 1)
            alpha = self.cal_alpha(self.Epochs, j + 1)
            # RDPSO 与 PSO 权重
            W = self.cal_W(self.Epochs, j + 1, self.yuzhi)
            # pso、rdpso、ardpso
            table = {'pso': self.velocity_update( self.v,self.population, self.pbest, self.gbest, w),
                     'rdpso': self.RDPSO_velocity_update(self.v,self.population, self.pbest, self.gbest, alpha,w),
                     'ardpso': W * self.velocity_update(self.v,self.population, self.pbest, self.gbest, w) + (1 - W) * self.RDPSO_velocity_update(self.v,self.population, self.pbest, self.gbest, alpha,w)
                     }
            # 更新速度
            self.v = table[self.key]
            # 更新位置
            self.population = self.position_update(self.population, self.v)
            # 计算每个粒子的目标函数和约束惩罚项
            self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
            self.conv = np.array([self.constraints(chrom) for chrom in self.population])  # 约束初始化
            # 更新个体历史最优位置
            for p in range(self.pop_size):
                if(self.fitness(self.population[p]) < self.fitness(self.pbest[p])):
                    self.pbest[p] = self.population[p] # 更新个体历史最优位置
            # 更新种群历史最优位置
            if np.min(self.fit) < self.gbest_fit:
                self.gbest =  self.population[np.argmin(self.fit)].copy() # 种群历史最优位置
                self.gbest_fit = np.min(self.fit)
            fitness.append(self.gbest_fit)
            fitness_mean.append(np.mean(self.fit))
            # 打印每次迭代的种群最优值，均值，最差值，方差
            print('epoch:', j, 'best:', self.gbest_fit,
                  'mean:', np.mean(self.fit),
                  'worst:', np.max(self.fit),
                  'std:', np.std(self.fit),
                  'NFE:', self.NFE,
                  'conv:', self.constraints(self.gbest))

            if count > 200:
                break
            if pre - (np.min(self.fit) + self.constraints(self.gbest)) < 1e-6:
                count += 1
            else:
                count = 0
            pre = np.min(self.fit) + self.constraints(self.gbest)
        return self.gbest,fitness,fitness_mean # 返回最优个体


    def calc_Lj(self,e1, e2, e3):
        """根据每个粒子的约束惩罚项计算Lj权重值，e1, e2列向量，表示每个粒子的第1个第2个约束的惩罚项值"""
        # 注意防止分母为零的情况
        # if (e1.sum() + e2.sum() + e3.sum()) <= 0:
        #     return 0, 0, 0
        # else:
        #     L1 = e1.sum() / (e1.sum() + e2.sum() + e3.sum())
        #     L2 = e2.sum() / (e1.sum() + e2.sum() + e3.sum())
        #     L3 = e3.sum() / (e1.sum() + e2.sum() + e3.sum())
        # # print('L1, L2:',L1, L2)
        L1, L2, L3 = 1,1,1
        return L1, L2, L3

    # 自适应权重 -- 线性方法
    def cal_w(self, iter_max, iter):
        # 自适应权重
        w_max = 2
        w_min = 0.4
        return w_max - (w_max - w_min) * iter / iter_max

    # 自适应权重 -- 线性方法
    def cal_alpha(self, iter_max, iter):
        # 自适应权重
        w_max = 0.9
        w_min = 0.3
        return w_max - (w_max - w_min) * iter / iter_max

    # 权重W -- 前期用pso，后期用rdpso
    def cal_W(self, iter_max, iter, yuzhi):
        # 权重
        w = 1
        if iter / iter_max >= yuzhi:
            w = 0
        return w

    def velocity_update(self,V, X, pbest, gbest, w):
        """
        根据速度更新公式更新每个粒子的速度
        :param V: 粒子当前的速度矩阵，self.pop_size*dim 的矩阵
        :param X: 粒子当前的位置矩阵，
        :param pbest: 每个粒子历史最优位置，针对某一个粒子来说
        :param gbest: 种群历史最优位置，1*dim 的矩阵
        """
        r1 = np.random.random((self.pop_size, 1))
        r2 = np.random.random((self.pop_size, 1))

        '''gbest_temp = 100*list(gbest)

        #print('gbest_temp',gbest_temp)
        print(len(gbest_temp))
        gbest_temp = np.array(gbest_temp).reshape(100, 21)'''
        # print(gbest.shape)
        V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)  # 直接对照公式写就好了

        # V = w * V + c1 * np.dot((pbest - X),r1) + c2 * np.dot((gbest - X),r2)  # 直接对照公式写就好了
        # 防止越界处理
        V[V < -self.max_vel] = -self.max_vel
        V[V > self.max_vel] = self.max_vel

        return V

    def RDPSO_velocity_update(self,V, X, pbest, gbest,alpha ,w):
        """
        根据速度更新公式更新每个粒子的速度
        :param V: 粒子当前的速度矩阵，self.pop_size*dim 的矩阵
        :param X: 粒子当前的位置矩阵，
        :param pbest: 每个粒子历史最优位置，self.pop_size*dim 的矩阵
        :param gbest: 种群历史最优位置，1*dim 的矩阵
        """
        # 计算VD
        φ = np.random.random((self.pop_size, 1))  # (0,1)随机数
        β = 1.45
        spbest = φ * pbest + (1 - φ) * gbest
        VD = β * (spbest - X)
        # 计算VR
        temp = sum(pbest[i] for i in range(len(pbest)))
        C = temp / len(pbest)
        λ = np.random.random((self.pop_size, 1))
        VR = alpha * abs(C - X) * λ
        # V = VR + VD
        V = w*V + VR + VD
        # V = w * V + c1  * r1*(pbest - X) + c2 * r2*(gbest- X)  # 直接对照公式写就好了


        # 防止越界处理
        V[V < -self.max_vel] = -self.max_vel
        V[V > self.max_vel] = self.max_vel

        return V

    def position_update(self,X, V):
        """
        根据公式更新粒子的位置
        :param X: 粒子当前的位置矩阵，维度是
        :param V: 粒子当前的速度举着，维度是
        """
        X = X + V  # 更新位置
        for i, j in itertools.product(range(self.pop_size), range(self.dim)):
            while (X[i][j]) < self.lowbound or (X[i][j]) > self.upbound:
                if(X[i][j] <= self.lowbound):
                    X[i][j] = self.lowbound
                else:
                    X[i][j] = self.upbound

            # 比例大于0且小于2的，置0
            # if (X[i][j] >= 0 and X[i][j] <= 5):
            #     X[i][j] = 0
        return X

    def update_pbest(self,pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):
        """
        判断是否需要更新粒子的历史最优位置
        :param pbest: 历史最优位置
        :param pbest_fitness: 历史最优位置对应的适应度值
        :param pbest_e: 历史最优位置对应的约束惩罚项
        :param xi: 当前位置
        :param xi_fitness: 当前位置的适应度函数值
        :param xi_e: 当前位置的约束惩罚项
        :return:
        """
        # 下面的 0.0000001 是考虑到计算机的数值精度位置，值等同于0
        # 规则1，如果 pbest 和 xi 都没有违反约束，则取适应度小的
        if pbest_e <= 0.0000001 and xi_e <= 0.0000001:
            if pbest_fitness <= xi_fitness:
                return pbest, pbest_fitness, pbest_e
            else:
                return xi, xi_fitness, xi_e
        # 规则2，如果当前位置违反约束而历史最优没有违反约束，则取历史最优
        if pbest_e < 0.0000001 and xi_e >= 0.0000001:
            return pbest, pbest_fitness, pbest_e
        # 规则3，如果历史位置违反约束而当前位置没有违反约束，则取当前位置
        if pbest_e >= 0.0000001 and xi_e < 0.0000001:
            return xi, xi_fitness, xi_e
        # 规则4，如果两个都违反约束，则取适应度值小的
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e

    def update_gbest(self,gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e):
        # sourcery skip: assign-if-exp, hoist-repeated-if-condition, remove-redundant-if
        """
        更新全局最优位置
        :param gbest: 上一次迭代的全局最优位置
        :param gbest_fitness: 上一次迭代的全局最优位置的适应度值
        :param gbest_e:上一次迭代的全局最优位置的约束惩罚项
        :param pbest:当前迭代种群的最优位置
        :param pbest_fitness:当前迭代的种群的最优位置的适应度值
        :param pbest_e:当前迭代的种群的最优位置的约束惩罚项
        :return:
        """
        # 先对种群，寻找约束惩罚项=0的最优个体，如果每个个体的约束惩罚项都大于0，就找适应度最小的个体
        pbest2 = np.concatenate([pbest, pbest_fitness.reshape(-1, 1), pbest_e.reshape(-1, 1)], axis=1)  # 将几个矩阵拼接成矩阵
        # print('pbest2=',pbest2)
        # print('pbest2[:, -1]=',pbest2[:, -1])
        pbest2_1 = pbest2[pbest2[:, -1] <= 0.0000001]  # 找出没有违反约束的个体
        if len(pbest2_1) > 0:
            pbest2_1 = pbest2_1[pbest2_1[:, self.dim].argsort()]  # 根据适应度值排序
        else:
            pbest2_1 = pbest2[pbest2[:, self.dim].argsort()]  # 如果所有个体都违反约束，直接找出适应度值最小的
        # 当前迭代的最优个体
        pbesti, pbesti_fitness, pbesti_e = pbest2_1[0, :self.dim], pbest2_1[0, self.dim], pbest2_1[0, self.dim + 1]
        # 当前最优和全局最优比较
        # 如果两者都没有约束
        if gbest_e <= 0.0000001 and pbesti_e <= 0.0000001:
            if gbest_fitness < pbesti_fitness:
                return gbest, gbest_fitness, gbest_e
            else:
                return pbesti, pbesti_fitness, pbesti_e
        # 有一个违反约束而另一个没有违反约束
        if gbest_e <= 0.0000001 and pbesti_e > 0.0000001:
            return gbest, gbest_fitness, gbest_e
        if gbest_e > 0.0000001 and pbesti_e <= 0.0000001:
            return pbesti, pbesti_fitness, pbesti_e
        # 如果都违反约束，直接取适应度小的
        if gbest_fitness < pbesti_fitness:
            return gbest, gbest_fitness, gbest_e
        else:
            return pbesti, pbesti_fitness, pbesti_e

    def cal_gongshi(self,Vars, B, M, t):
        # 计算工时函数
        # global gongShi # 工时=比例*订单加工总数
        gongShi = [] * B * M
        temp = 0
        for i in range(B):
            tem = [0] * M  # 临时子工时
            tem[i] = np.multiply(Vars[0 + temp:M + temp], t[i])
            gongShi.extend(tem[i])
            temp += M
        return gongShi

if __name__ == '__main__':
    from od_f import calc_e1, calc_e2, calc_e3, calc_e4, fitness, constraints, rt_idm, task_name

    lowwer = 0
    upper = 100
    pop_size = 300
    dim = rt_idm
    epochs = 500

    good_p=np.array( [44.28594868536934, 29.315631301093028, 5.091229367497709, 9.27332840045514, 1.0527861237904335, 9.356935940884295, 1.6241401809100566, 0.37698132516230853, 7.77278220421577, 5.189158081884436, 2.751433871790157, 49.4321196628027, 7.58363325484447, 26.893891599300158, 16.581367222235695, 15.142607286601967, 1.8497898754506013, 13.757579275349782, 33.13625573911796, 12.754410775317954, 6.777989825926031, 8.006537481113211, 38.20838970052688, 12.149400464728382, 12.54011181482722, 4.85948826499084, 22.615684510746313, 1.620387763067151, 34.90295939555978, 2.022153777456961, 2.81852680007362, 14.273581028010446, 11.500302495648176, 25.840465600904622, 8.642010902346387, 4.684481661967465, 16.653952110557835, 4.5452844459222455, 24.06037156655191, 5.2054810734778485, 10.886861210755702, 33.96356793076698, 8.570852296158126, 3.6453091581744026, 3.7388598534832576, 4.801150738790935, 28.65248024431379, 18.001693197169686, 32.589654511909785, 1.3565196132204158, 2.419695877109211, 1.943937675082145, 8.375360483806189, 59.70757039507218, 7.538781553849088, 18.65813440186077, 18.61445434483161, 6.512110317678458, 9.400737268585289, 3.914136207683804, 0.47738624897084925, 60.89029430837846, 0.19088130387154403] )

    print(fitness(good_p))
    print(calc_e1(good_p))
    print(calc_e2(good_p))
    print(calc_e3(good_p))
    print(calc_e4(good_p))

    # good_p 参数
    sample_size = 20
    stdev_data = 1
    # ARDPSO_MY 参数
    key = 'ardpso'
    # dim = M * O * Oir  # 粒子的维度     改 7*3*3
    # dim = 10 # 粒子的维度     改 7*3*3
    ardpso = ARDPSO(fitness, constraints,lowwer, upper, pop_size, dim, key, epochs,sample_size,good_p,stdev_data)
    # 获取当前时间
    now = '%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(ardpso.__class__.__name__ + '_' + now)
    import xlwt

    file = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = file.add_sheet('fit_best_ok', cell_overwrite_ok=True)  # 创建sheet工作表
    sheet2 = file.add_sheet('fit_best_all', cell_overwrite_ok=True)  # 创建sheet工作表
    sheet3 = file.add_sheet('fit_mean_ok', cell_overwrite_ok=True)  # 创建sheet工作表
    sheet4 = file.add_sheet('fit_mean_all', cell_overwrite_ok=True)  # 创建sheet工作表
    for i in range(1):
        # 获取开始时间
        start = time.perf_counter()
        best,fit,fit_mean = ardpso.run()
        print("ardpso.yuzhi = ",ardpso.yuzhi)
        print(best)
        print(fitness(best))
        # print(con(best))
        print(calc_e1(best))
        print(calc_e2(best))
        print(calc_e3(best))
        print(calc_e4(best))
        # 获取结束时间
        end = time.perf_counter()
        # 计算运行时间
        runTime = end - start
        # 输出运行时间
        print("运行时间：", runTime, "秒")
        # # 保存结果
        for x in range(len(fit)):
            sheet2.write(x, i, fit[x])  # 写入数据参数对应 行, 列, 值
        sheet2.write(len(fit) + 1, i, runTime)  # 写入数据参数对应 行, 列, 值
        for x in range(len(fit_mean)):
            sheet4.write(x, i, fit_mean[x])  # 写入数据参数对应 行, 列, 值
        sheet4.write(len(fit_mean) + 1, i, runTime)  # 写入数据参数对应 行, 列, 值
        if (calc_e1(best) == 0 and calc_e2(best) == 0.0 and calc_e3(best) == 0 and calc_e4(best) == 0):
            # 保存结果
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save information in log.txt
            f_dir = os.path.join(save_dir, 'log.txt')
            f = open(f_dir, 'a+')
            f.write('pop_size = ' + str(pop_size) + '\n' +
                    'epochs = ' + str(epochs) + '\n' +
                    'sample_size =' + str(sample_size) + '\n' +
                    'stdev_data =' + str(stdev_data) + '\n' +
                    '运行时间 =' + str(runTime) + "秒" + '\n' +

                    'now_time: ' + now + '\n' +
                    'fit =' + str(fit) + '\n' +
                    'best= ' + str(best) + '\n')
            f.close()
            # 保存结果
            for x in range(len(fit)):
                sheet1.write(x, i, fit[x])  # 写入数据参数对应 行, 列, 值
            sheet1.write(len(fit) + 1, i, runTime)  # 写入数据参数对应 行, 列, 值
            for x in range(len(fit_mean)):
                sheet3.write(x, i, fit_mean[x])  # 写入数据参数对应 行, 列, 值
            sheet3.write(len(fit_mean) + 1, i, runTime)  # 写入数据参数对应 行, 列, 值
            plt.xlabel('迭代次数')
            plt.ylabel('适应度')
            plt.title('迭代过程')
            plt.plot(fit[: epochs], color='r')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'fitness-new.png'), bbox_inches='tight', dpi=600)
            plt.close()
            # plt.show()
    file.save(save_dir + '.xls')  # 保存.xls到当前工作目录




