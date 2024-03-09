'''
订单分配约束函数
'''
import numpy as np
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
        e1[i] = (np.abs(X[temp:M + temp].sum() - 100)) * 100000
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
                e2 = (amin[j + temp] - np.multiply(t[i], X[j + temp])) * 100000  # 惩罚e2=差值
                # X[j+temp] = -X[j+temp]
            # X[j] = 0
            elif np.multiply(t[i], X[j + temp]) >= amax[j + temp]:
                e2 = (np.multiply(t[i], X[j + temp]) - amax[j + temp]) * 100000
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

    e3 = abs(ct_max - Co[0]) * 100000
    if abs(ct_max - Co) < 0.1:
        return 0
    return e3