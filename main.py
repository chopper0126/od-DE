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
    from od_f import calc_e1,calc_e2,calc_e3,calc_e4,fitness,constraints,rt_idm,task_name,good_p

    lowwer = 0
    upper = 100
    pop_size = 300
    dim = rt_idm
    epochs = 500
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
    # list = [ardpso]
    list = [jade,shade,lshade]
    fit_mean = []
    for j in range(len(list)):
        # 获取当前时间
        now = '%s' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        save_dir = os.path.join(list[j].__class__.__name__ + '_' +task_name+'_'+ now)
        import xlwt
        file = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
        sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
        for i in range(1):
            if list[j].__class__.__name__ == 'PSO' or list[j].__class__.__name__ == 'RDPSO' or list[j].__class__.__name__ == 'ARDPSO':
                best,fit ,fit_mean = list[j].run()
            else:
                best, fit = list[j].run()
            print(list[j])
            print(best)
            # print(fit)
            print(fitness(best))
            print(calc_e1(best))
            print(calc_e2(best))
            print(calc_e3(best))
            print(calc_e4(best))
            if (calc_e1(best) == 0 and calc_e2(best) == 0.0 and calc_e3(best) == 0 and calc_e4(best) == 0):
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
                        'best = ' + str(best.tolist()) + '\n')
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




