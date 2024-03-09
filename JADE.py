from math import sin, e

import numpy as np
from numpy import cos, pi


class JADE():
    '''
    JADE算法：根据论文JADE: Adaptive Differential Evolution withOptional External Archive复现
    特点：   1.采用自适应策略，自适应缩放因子F和自适应交叉概率CR，将成功的CR和F存储进SCR和SF中，然后根据该集合的均值更新uCR和uF，SCR和SF每代都会重置更新
            2.采用外部存档，存储每次迭代中的遗弃个体
            3.采用变异策略DE/current-to-pbest/1
    输入：   fitness:适应度函数
            constraints:约束条件
            lowwer:下界
            upper:上界
            pop_size:种群大小
            dim:维度
            mut_way:变异方式
            epochs:迭代次数
    输出：   best:最优个体

    本程序在原JADE算法上增加约束处理策略，使其能使用于约束优化问题
    '''

    def __init__(self, fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs):
        self.fitness = fitness  # 适应度函数
        self.constraints = constraints  # 约束条件
        self.lowbound = lowwer  # 下界
        self.upbound = upper  # 上界
        self.pop_size = pop_size  # 种群大小
        self.dim = dim  # 种群大小
        self.population = np.random.rand(self.pop_size, self.dim)  # 种群
        self.best = self.population[0]  # 最优个体
        self.fit=np.random.rand(self.pop_size)#适应度
        self.conv=np.random.rand(self.pop_size)#约束
        self.mut_way = mut_way#变异策略
        self.Archive = []  # 存档
        #*********此部分参数为CR和F相关参数*********
        self.uCR = 0.5  # CR的期望
        self.uF = 0.5  # F的期望
        self.F = self.uF + np.sqrt(0.1) * np.random.standard_cauchy(self.pop_size)  # 缩放因子
        self.CR = np.random.normal(self.uCR, 0.1, self.pop_size)  # 交叉概率
        self.c = 0.02  # 系数
        self.SCR = []  # 存放成功的CR
        self.SF = []  # 存放成功的F
        #***************************************
        self.Epochs = epochs  # 迭代次数
        self.NFE = 0  # 记录函数调用次数
        self.max_NFE = 10000*self.dim  # 最大函数调用次数

    #初始化种群
    def initpop(self):
        self.population = self.lowbound + (self.upbound - self.lowbound) * np.random.rand(self.pop_size,
                                                                                          self.dim)  # 种群初始化
        self.fit = np.array([self.fitness(chrom) for chrom in self.population])  # 适应度初始化,此处复杂度为pop_size
        self.conv = np.array([self.constraints(chrom) for chrom in self.population])  # 约束初始化
        self.NFE += self.pop_size  # 记录函数调用次数
        self.best = self.population[np.argmin(self.fit)]  # 最优个体初始化
    #变异操作
    def mut(self):
        mut_population = []  # 定义新种群
        if self.mut_way == 'DE/current-to-pbest/1':
            for i in range(self.pop_size):
                p = 0.05
                # 选择适应度值前p的个体
                idx_set = np.argsort(self.fit)[:int(self.pop_size * p)]
                xpbest_idx = np.random.choice(idx_set, 1, replace=False)
                xpbest = self.population[xpbest_idx].flatten()  # 因为argsort返回值作索引时会变成二维，所以要flatten
                a = self.population[
                    np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)].flatten()
                # b为外部存档和当前种群的并集中随机选择的个体
                idxb = np.random.choice(np.arange(self.pop_size + len(self.Archive)), 1, replace=False)[0]  # 随机选择一个下标
                if (idxb >= self.pop_size):  # 如果下标大于种群大小，则从存档中选择
                    idxb -= self.pop_size
                    b = self.Archive[idxb]
                else:  # 否则从当前种群中选择
                    b = self.population[idxb].flatten()
                v = self.population[i] + self.F[i] * (xpbest - self.population[i]) + self.F[i] * (a - b)  # 突变运算
                mut_population.append(v)
        return mut_population
    #交叉操作
    def cross(self, mut_population):
        cross_population = self.population.copy()
        #Binomial crossover
        for i in range(self.pop_size):
            j = np.random.randint(0, self.dim)  # 随机选择一个维度
            for k in range(self.dim):  # 对每个维度进行交叉
                if np.random.rand() < self.CR[i] or k == j:  # 如果随机数小于交叉率或者维度为j
                    cross_population[i][k] = mut_population[i][k]  # 交叉
                    #边界处理
                    if cross_population[i][k]>self.upbound:
                        cross_population[i][k]=(self.upbound+self.population[i][k])/2#如果超过上界，则取上界和原来的中间值
                    elif cross_population[i][k]<self.lowbound:
                        cross_population[i][k]=(self.lowbound+self.population[i][k])/2
        return cross_population
    #选择操作
    def select(self, cross_population):
        self.SF = []  # 每一代都要清空
        self.SCR = []
        for i in range(self.pop_size):#此处复杂度为pop_size
            temp=self.fitness(cross_population[i])
            temp_v=self.constraints(cross_population[i])
            self.NFE += 1  # 记录函数调用次数
            if (self.conv[i]==0 and temp_v==0 and temp<self.fit[i]) or (temp_v<self.conv[i]):#如果新个体适应度和约束都优于原来的
                self.population[i] = cross_population[i]
                self.fit[i] = temp
                self.conv[i] = temp_v
                self.SCR.append(self.CR[i])
                self.SF.append(self.F[i])
                self.Archive.append(self.population[i])
                if len(self.Archive) > self.pop_size * 2.6:
                    # 如果存档超过种群大小，则随机删除一些
                    for o in range(int(len(self.Archive) - self.pop_size * 2.6)):
                        self.Archive.pop(np.random.randint(0, len(self.Archive)))
    #迭代
    def run(self):
        pre=np.min(self.fit)
        count=0
        pre=0
        self.initpop()  # 初始化种群
        for i in range(self.Epochs):  # 迭代
            #每一代开始时，生成F和CR
            self.F = np.clip(self.uF + np.sqrt(0.1) * np.random.standard_cauchy(self.pop_size), 0.000001,1)  # 缩放因子更新
            self.CR = np.clip(np.random.normal(self.uCR, 0.1, self.pop_size), 0, 1)  # 交叉概率更新
            #变异操作，使用DE/current-to-pbest/1策略
            mut_population = self.mut()  # 变异
            #交叉操作
            cross_population = self.cross(mut_population)  # 交叉
            #选择操作，并更新历史归档A和SCR，SF
            self.select(cross_population)  # 选择
            self.best = self.population[np.argmin(self.fit)]  # 更新最优个体
            #每一代结束时更新uCR和uF
            self.uCR = (1 - self.c) * self.uCR + self.c * np.mean(self.SCR)  # 更新平均交叉概率
            self.uF = (1 - self.c) * self.uF + self.c * np.sum(np.array(self.SF) ** 2) / np.sum(self.SF)  # 更新平均缩放因子
            # 打印每次迭代的种群最优值，均值，最差值，方差
            print('epoch:', i, 'best:', np.min(self.fit),
                  'mean:',np.mean(self.fit),
                  'worst:',np.max(self.fit),
                  'std:',np.std(self.fit),
                  'NFE:',self.NFE,
                  'conv:',self.constraints(self.best))
            # if self.NFE > self.max_NFE:
            #     break
            if count>200:
                break
            if pre-(np.min(self.fit)+self.constraints(self.best))<1e-6:
                count+=1
            else:
                count=0
            pre=np.min(self.fit)+self.constraints(self.best)
        return self.best  # 返回最优个体


if __name__ == '__main__':
    # def fitness(x):
    #     y = 0
    #     for i in range(len(x)):
    #         y = y + x[i] ** 2 - 10 * cos(2 * pi * x[i]) + 10
    #
    #     return y


    # def constraints(x):
    #     g1 = 0
    #     g2 = 0
    #     for i in range(len(x)):
    #         g1 = g1 + (-x[i] * sin(2 * x[i]))
    #         g2 = g2 + x[i] * sin(x[i])
    #     g = max(g1, 0) + max(g2, 0)
    #     return g/2

    from get_data import init_data
    M, B, d, P, t, yunfei, amin, amax ,Co ,Pir ,mean_v = init_data()


    def fitness(X):
        # 生成距离决策矩阵
        global D2
        D2 = []
        temp = 0
        for _ in range(B):
            for j in range(M):
                if X[j + temp] > 0:
                    D2.append(1)
                else:
                    D2.append(0)
            temp += M

        # define list
        D2 = np.asarray(D2)
        D2 = np.array(D2, dtype=int)
        global gongShi  # 工时=比例*订单加工总数
        gongShi = [] * B * M
        temp = 0
        for i in range(B):
            tem = [0] * M  # 临时子工时
            tem[i] = np.multiply(X[0 + temp:M + temp], t[i])
            gongShi.extend(tem[i])
            temp += M

        # print("优化后工时",gongShi)
        f1 = np.dot(gongShi, P[:B * M])
        f2 = np.dot(D2, d[:B * M])

        f = f1 + f2 * yunfei * 100  # 计算目标函数值, 生产费用 = 工时*报价，运输费用=单位距离运费*距离决策矩阵*距离
        f = f / 100
        # print('订单分配工时:',gongShi)
        # 计算每个订单的成本
        global everyBcost
        everyBcost = [-1] * B
        temp = 0
        for i in range(B):
            everyBcost[i] = np.dot(gongShi[0 + temp:M + temp], P[0 + temp:M + temp]) + yunfei * np.dot(
                D2[0 + temp:M + temp], d[0 + temp:M + temp]) * 100
            temp += M
        # 计算每个供应商的成本
        global everyMcost
        everyMcost = [-1] * B * M
        for i in range(B * M):
            everyMcost[i] = np.dot(gongShi[i], P[i]) + yunfei * np.dot(D2[i], d[i]) * 100
        # print('适应度',f)
        return f


    def calc_e1(X):
        """计算第一个约束的惩罚项"""

        """计算群体惩罚项，X 的维度是 size * 8"""
        # sumcost=[]

        # 等式惩罚项系数
        # 对每一个个体
        e1 = [0] * B
        """计算第一个约束的惩罚项   等式约束"""
        '''e1[0] = np.abs(X[:8].sum()- 100) #第一行加起来如果不是100就惩罚
        e1[1] = np.abs(X[8:16].sum()- 100)
        E1.append(e1)'''
        temp = 0
        for i in range(B):
            e1[i] = (np.abs(X[temp:M + temp].sum() - 100))
            temp += M

        # e_zong = np.abs(X.sum()- 100*B)

        if abs(sum(e1) - 100 * B) < 0.1 and all(abs(j - 100) < 0.1 for j in e1):
            print('dengshi', 0)
            return 0

        # e = X[0] + X[1] - 6
        ds_error1 = sum(e1)
        # print('等式约束',e1)
        return ds_error1 * 1

    def calc_e2(X):
        """计算第二个约束的惩罚项"""
        # 新增一个订单，相当于加了7个不等式约束
        # 前提条件，只有A1-A3三种资源，供应商表不变。
        # 想法1：订单工时t，按照资源类型排，加个判断
        # 已测试 增加一个订单B4，且资源类型为A1的情况
        """计算第二个约束的惩罚项   不等式约束"""
        e2 = 0
        ee = [0] * B
        temp = 0
        e_a1 = 0
        e_a2 = 0
        e_a3 = 0
        for i in range(B):
            for j in range(M):
                if np.multiply(t[i], X[j + temp]) <= amin[j + temp]:  # 如果该加工时数小于规定最小值 bv
                    e2 = (amin[j + temp] - np.multiply(t[i], X[j + temp]))   # 惩罚e2=差值
                    # X[j+temp] = -X[j+temp]
                # X[j] = 0
                elif np.multiply(t[i], X[j + temp]) >= amax[j + temp]:
                    e2 = (np.multiply(t[i], X[j + temp]) - amax[j + temp])
                # E2.append(e2)
                ee[i] += e2
            temp += M
        bds_e = (sum(ee) + e_a1 + e_a3 + e_a2)
        return bds_e / 100

    def calc_e3(X):
        '''时间约束惩罚项'''
        e3 = 0
        # 生成距离决策矩阵
        global D2
        D2 = []
        temp = 0
        for _ in range(B):
            for j in range(M):
                if X[j + temp] > 0:
                    D2.append(1)
                else:
                    D2.append(0)
            temp += M

        # define list
        D2 = np.asarray(D2)
        D2 = np.array(D2, dtype=int)
        global gongShi  # 工时=比例*订单加工总数
        gongShi = [] * B * M
        temp = 0
        for i in range(B):
            tem = [0] * M  # 临时子工时
            tem[i] = np.multiply(X[0 + temp:M + temp], t[i])
            gongShi.extend(tem[i])
            temp += M

        DT = np.divide(D2 * d[:B * M], mean_v)

        DT = np.multiply(DT, 100)
        CT = DT + gongShi
        CT = np.asarray(CT)
        ct_max = np.max(CT) / 100

        e3 = abs(ct_max - Co[0])
        if abs(ct_max - Co) < 0.1:
            return 0
        return e3

    def constraints(x):
        return 3*calc_e1(x) + calc_e2(x) + 10*calc_e3(x)



    lowwer = 0
    upper = 100
    pop_size = 300
    dim = 21
    mut_way = 'DE/current-to-pbest/1'
    epochs = 1500
    jade = JADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    best = jade.run()
    print(best)
    print(fitness(best))
