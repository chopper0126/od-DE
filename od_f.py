import numpy as np
import  get_data
from get_data import init_data, init_data_e2, init_data_e7
# M, Oir, O, d, P, t, yunfei, amin, amax, Co, Pir, mean_v ,good_p,E,EC= init_data()
M, Oir, O, d, P, t, yunfei, amin, amax, Co, Pir, mean_v ,good_p,E,EC= get_data.init_data()
task_name = get_data.init_data_e5.__name__[-2:]


l1, l2, l3 ,l4= 10000,1000,100,1
rt_idm = M*Oir*O
def fitness(X):
    '''
    目标函数
    '''
    # 生成距离决策矩阵
    global D2
    D2 = []
    for x in X:
        if x > 0:
            D2.append(1)
        else:
            D2.append(0)
    # define list
    D2 = np.asarray(D2)
    D2 = np.array(D2, dtype=int)
    global gongShi  # 工时=比例*订单加工总数
    gongShi = [] * O * M * Oir
    temp = 0
    temp2 = 0
    for i in range(O):
        for j in range(Oir):
            tem = [0] * M  # 临时子工时
            tem[i] = np.multiply(X[0 + temp:M + temp], t[j + temp2])
            gongShi.extend(tem[i])
            temp += M
        temp2 += Oir
    f1 = np.dot(gongShi, P[:O * M * Oir])
    f2 = np.dot(D2, d[:O * M * Oir])
    f = f1 + f2 * yunfei * 100  # 计算目标函数值, 生产费用 = 工时*报价，运输费用=单位距离运费*距离决策矩阵*距离
    f = f / 100
    f = all_e(X, l1, l2, l3,l4) + f
    # print('f = ', f)
    return f

def calc_e1(X):
    """
    计算第一个约束的惩罚项
    计算群体惩罚项，X 的维度是 size * 8
    等式约束：每个订单工序分配给供应商的比例总数等于100
    """
    e1 = [0] * O * Oir
    temp = 0
    temp2 = 0
    for i in range(O):
        for j in range(Oir):
            e1[j + temp2] = (np.abs(X[temp:M + temp].sum() - 100)) * 1
            if (np.abs(X[temp:M + temp].sum() - 100) < 1):
                e1[j + temp2] = 0
            temp += M
        temp2 += Oir
    ds_error1 = sum(e1)
    # print('e1:', e1)
    return ds_error1

def calc_e2(X):
    """
    计算第二个约束的惩罚项
    不等式约束：分配的工时，不能超过供应商剩余能力上下限
    """
    e2 = []
    temp = 0
    count = O
    for k in range(Oir):
        for j in range(M):
            r_gongshi_sum = 0
            while (count > 0):
                r_gongshi_sum += np.multiply(t[k + O * (count - 1)], X[j + M * Oir * (count - 1) + temp])
                count -= 1
            count = O
            if (r_gongshi_sum) >= amax[j + temp]:
                # 如果该加工时数大于于规定最大值 bv
                e2.append(1 * (r_gongshi_sum - amax[j + temp]))
            else:
                e2.append(0)
        temp += M

    bds_e = sum(e2)
    # print('e2:', e2,"sum(e2):" ,bds_e/100 )
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
    for i in range(O):
        for _ in range(Oir):
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
    gongShi = [] * O * M * Oir
    temp = 0
    temp2 = 0
    for i in range(O):
        for j in range(Oir):
            tem = [0] * M  # 临时子工时
            tem[i] = np.multiply(X[0 + temp:M + temp], t[j + temp2])
            gongShi.extend(tem[i])
            temp += M
        temp2 += Oir
    # DT 运输时间 = 距离/平均速度
    DT = np.divide(D2 * d[:O * M * Oir], mean_v)
    DT = np.multiply(DT, 100)

    CT = DT + gongShi
    CT = np.asarray(CT)
    ct_max = [0] * O
    e3 = [0] * O
    for i in range(O):
        ct_max[i] = np.max(CT[Oir * M * i:Oir * M * (i + 1)]) / 100
        if (Co[i] - ct_max[i] >= 0):
            e3[i] = 0
        else:
            e3[i] = (ct_max[i] - Co[i]) * 1

    # print('e3:', e3)
    # if all(i < 0.1 for i in e3):
    #     return 0
    return sum(e3)

def calc_e4(X):
    """
    计算第四个个约束的惩罚项  能耗约束惩罚项
    不等式约束：能源消耗 不能超过订单总能源限制E
    """
    e4 = []
    temp = 0
    temp2 = 0
    count = O
    "计算每个订单的能耗"
    for i in range(O):
        "计算某个订单所有工序的能耗"
        r_e_sum = 0
        for k in range(Oir):
            "计算某个订单工序分配给不同供应商的能耗"
            for j in range(M):
                r_e_sum += t[k + temp2]*X[j+ M * Oir * i + temp]*EC[j+temp]
            temp += M
        temp2 += Oir

        if (r_e_sum) >= E[i]:
            # 如果该加工时数大于于规定最大值 bv
            e4.append(1 * (r_e_sum - E[i]))
        else:
            e4.append(0)
        temp = 0


    bds_e = sum(e4)
    # print('e2:', e2,"sum(e2):" ,bds_e/100 )
    return bds_e


def all_e(x, l1, l2, l3,l4):
    '''
    罚函数系数
    :param x:
    :return:
    '''
    return l1 * calc_e1(x) + l2 * calc_e2(x) + l3 * calc_e3(x) + l4 *calc_e4(x)

def constraints(x):
    return 0

if __name__ == '__main__':
    # 比例
    best = np.array([4.97360385,0.,        0.,       12.55654519 ,11.02385107, 67.61493321,
            2.83115202,  0. ,         0. ,         0.  ,       16.96413756 ,13.04611669,
            2.21769222, 66.84686075,  0.  ,       20.21834556  ,0.  ,      15.07652622,
            0.     ,     0.   ,      63.89338946 , 0.  ,       26.17919626, 13.20041585,
            41.13799475 , 0.  ,        1.85313595 ,16.63429463 , 4.57299957, 23.33642237,
            16.42990332 ,22.24180971 ,21.18716574 , 3.51285019  ,7.72240651,  0.,
            5.06083996 ,26.44232638 ,55.02778997 , 0.  ,        0.  ,       12.47008365,
            2.53165374 ,17.65605826, 12.05104221 ,18.64872213 , 8.10392712 ,13.90080396,
            26.10991965 ,15.79174625 , 2.27612575 , 2.4876844, 45.92692969 ,17.87284808,
            5.3350201 ,  9.31602853,  6.19689485,  0.   ,       0.     ,    15.94294132,
            54.87471204 , 8.19675592 ,13.78981104])

    print(fitness(best))
    print(calc_e1(best))
    print(calc_e2(best))
    print(calc_e3(best))
    print(calc_e4(best))
