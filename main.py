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

from ARDPSO import ARDPSO
from DE import DE
from JADE import JADE
from LSHADE import LSHADE
from PSO import PSO
from RDPSO import RDPSO
from SHADE import SHADE
from SaDE import SaDE
from ardpso_ex_canshu import ardpso_ex_canshu

import itertools
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
if __name__ == '__main__':
    from get_data import init_data, init_data_m100
    # M, Oir, O, d, P, t, yunfei, amin, amax, Co, Pir, mean_v, good_p = init_data()
    M, Oir, O, d, P, t, yunfei, amin, amax, Co, Pir, mean_v, good_p =init_data_m100

    #
    def fitness(X):
        '''
        目标函数
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

        # print("优化后工时",gongShi)
        f1 = np.dot(gongShi, P[:O * M * Oir])
        f2 = np.dot(D2, d[:O * M * Oir])
        D2_S = [[0 for i in range(Oir)]for i in range(M)] # 7 * 3 矩阵，供应商矩阵
        new_D2 = D2
        for i in range(3):
            for j in range(7):
                D2_S[j][i] = int(D2[j])




        f = f1 + f2 * yunfei * 100  # 计算目标函数值, 生产费用 = 工时*报价，运输费用=单位距离运费*距离决策矩阵*距离
        f = f / 100
        # print('订单分配工时:',gongShi)
        # # 计算每个订单的成本
        # global everyBcost
        # everyBcost = [-1] * O
        # temp = 0
        # for i in range(O):
        #     everyBcost[i] = np.dot(gongShi[0 + temp:M + temp], P[0 + temp:M + temp]) + yunfei * np.dot(
        #         D2[0 + temp:M + temp], d[0 + temp:M + temp]) * 100
        #     temp += M
        # # 计算每个供应商的成本
        # global everyMcost
        # everyMcost = [-1] * O * M
        # for i in range(O * M):
        #     everyMcost[i] = np.dot(gongShi[i], P[i]) + yunfei * np.dot(D2[i], d[i]) * 100
        # print('适应度',f)
        f = all_e(X) + f
        return f

    def calc_e1(X):
        """
        计算第一个约束的惩罚项
        计算群体惩罚项，X 的维度是 size * 8
        等式约束：每个订单工序分配给供应商的比例总数等于100
        """
        # 等式惩罚项系数
        # 对每一个个体
        e1 = [0] * O * Oir
        temp = 0
        temp2 = 0
        for i in range(O):
            for j in range(Oir):
                e1[j+temp2] = (np.abs(X[temp:M + temp].sum() - 100))
                if(np.abs(X[temp:M + temp].sum() - 100) < 1):
                    e1[j + temp2] = 0
                temp += M
            temp2 += Oir
        if abs(sum(e1) - 100 * O * Oir) < 1 and all(abs(j - 100) < 1 for j in e1):
            # print('dengshi', 0)
            return 0
        ds_error1 = sum(e1)
        # print('e1:',e1)
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
            if(Co[i] - ct_max[i] >= 0):
                e3[i] = 0
            else:
                e3[i] = ct_max[i] - Co[i]

        # print('e3:', e3)
        if all(i < 0.1 for i in e3):
            return 0
        return sum(e3)


    def calc_Lj(e1, e2, e3):
        """根据每个粒子的约束惩罚项计算Lj权重值，e1, e2,e3，表示每个粒子的第1个第2个 第三个约束的惩罚项值"""
        # 注意防止分母为零的情况
        if (e1 + e2+ e3) <= 0:
            return 0, 0, 0
        else:
            L1 = e1 / (e1+ e2 + e3)
            L2 = e2 / (e1+ e2 + e3)
            L3 = e3 / (e1+ e2 + e3)
        # print('L1, L2:',L1, L2)
        return L1, L2, L3
    def all_e(x):
        # L1, L2, L3 = calc_Lj(calc_e1(x),calc_e2(x) ,calc_e3(x) )
        # return (L1*calc_e1(x) + L2*calc_e2(x) + L3*calc_e3(x))
        return 1000*calc_e1(x) + 100*calc_e2(x) + calc_e3(x)

    def constraints(x):
        return 0

    lowwer = 0
    upper = 100
    pop_size = 300
    dim = M * O * Oir
    epochs = 1500
    mut_way_1 = 'DE/rand/1'
    mut_way = 'DE/current-to-pbest/1'
    jade = JADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    de = DE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way_1, epochs)
    sade = SaDE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way_1, epochs)
    # numpy 版本问题导致不能运行  numpy==1.23.5 可以运行
    shade = SHADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)
    lshade = LSHADE(fitness, constraints, lowwer, upper, pop_size, dim, mut_way, epochs)

    key, sample_size, stdev_data = ardpso_ex_canshu()
    ardpso = ARDPSO(fitness, constraints,lowwer, upper, pop_size, dim, key, epochs,sample_size,good_p,stdev_data)

    pso = PSO(fitness, constraints,lowwer, upper, pop_size, dim, epochs,sample_size,good_p,stdev_data)
    rdpso = RDPSO(fitness, constraints,lowwer, upper, pop_size, dim,epochs,sample_size,good_p,stdev_data)

    list = [ardpso,pso,rdpso,jade,shade,lshade]
    for j in range(len(list)):
        # 获取当前时间
        now = '%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        save_dir = os.path.join(list[j].__class__.__name__ + '_' + now)
        import xlwt
        file = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
        sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
        for i in range(1):
            best,fit = list[j].run()
            print(list[j])
            print(best)
            # print(fit)
            # print(fitness(best))
            print(calc_e1(best))
            print(calc_e2(best))
            print(calc_e3(best))
            if (calc_e1(best) == 0 and calc_e2(best) == 0.0 and calc_e3(best) == 0):
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # save information in log.txt
                f_dir = os.path.join(save_dir, 'log.txt')
                f = open(f_dir, 'a+')
                f.write('pop_size = ' + str(pop_size) + '\n' +
                        'epochs = ' + str(epochs) + '\n' +
                        'dim = ' + str(dim) + '\n' +
                        'now_time: ' + now + '\n' +
                        'fit =' + str(fit)+ '\n' +
                        'best = ' + str(best) + '\n')
                f.close()

                # 保存结果
                for x in range(len(fit)):
                    sheet1.write(x, i, fit[x])  # 写入数据参数对应 行, 列, 值

                plt.xlabel('迭代次数')
                plt.ylabel('适应值')
                plt.title('迭代过程')
                plt.plot(fit[: epochs], color='r')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'fitness-new.png'), bbox_inches='tight', dpi=600)
                plt.close()
                # plt.show()
        file.save(save_dir + '.xls')  # 保存.xls到当前工作目录




