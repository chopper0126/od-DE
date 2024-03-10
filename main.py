'''
主程序，用来跑所有算法的对比

目标函数：fitness
约束条件：constraints
数据：data
相同

算法：algorithm  不同的算法，包括PSO、RDPSO、ARDPSO、DE、JaDE、LSHADE、SaDE、SHADE等

输出：
'''
import  numpy as np

from DE import DE
from JADE import JADE
from LSHADE import LSHADE
from SHADE import SHADE
from SaDE import SaDE

if __name__ == '__main__':
    from get_data import init_data
    M, B, d, P, t, yunfei, amin, amax ,Co ,Pir ,mean_v = init_data()

    def fitness(X):
        '''
        目标函数
        '''
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
        """
        计算第一个约束的惩罚项
        计算群体惩罚项，X 的维度是 size * 8
        等式约束：每个订单工序分配给供应商的比例总数等于100
        """
        # 等式惩罚项系数
        # 对每一个个体
        e1 = [0] * B
        temp = 0
        for i in range(B):
            e1[i] = (np.abs(X[temp:M + temp].sum() - 100))
            temp += M
        if abs(sum(e1) - 100 * B) < 0.1 and all(abs(j - 100) < 0.1 for j in e1):
            print('dengshi', 0)
            return 0
        ds_error1 = sum(e1)
        return ds_error1 * 1

    def calc_e2(X):
        """
        计算第二个约束的惩罚项
        不等式约束：分配的工时，不能超过供应商剩余能力上下限

        关于增加订单的情况（暂时不涉及）
        新增一个订单，相当于加了7个不等式约束
        前提条件，只有A1-A3三种资源，供应商表不变。
        想法1：订单工时t，按照资源类型排，加个判断
        已测试 增加一个订单B4，且资源类型为A1的情况
        """
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
                elif np.multiply(t[i], X[j + temp]) >= amax[j + temp]:
                    e2 = (np.multiply(t[i], X[j + temp]) - amax[j + temp])
                ee[i] += e2
            temp += M
        bds_e = (sum(ee) + e_a1 + e_a3 + e_a2)
        return bds_e / 100

    def calc_e3(X):
        '''
        计算第三个个约束的惩罚项  时间约束惩罚项
        不等式约束：加工时间+运输时间 不能超过订单周期Co
        '''
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
        # DT 运输时间 = 距离/平均速度
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
    epochs = 1000
    mut_way_1 = 'DE/rand/1'
    mut_way = 'DE/current-to-pbest/1'
    jade = JADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    de = DE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way_1, epochs)
    sade = SaDE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way_1, epochs)
    shade = SHADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    lshad = LSHADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    list = [de,jade,shade,sade,lshad]
    for i in range(len(list)):
        best = list[i].run()
        print(list[i])
        print(best)
        print(fitness(best))




