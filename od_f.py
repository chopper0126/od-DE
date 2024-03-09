'''
订单分配目标函数：od_f
'''
import numpy as np
def calc_f(X):
    #生成距离决策矩阵
    global D2
    D2 =[]
    temp = 0
    for _ in range(B):
        for j in range(M):
            if X[j+temp] > 0:
                D2.append(1)
            else:
                D2.append(0)
        temp += M

# define list
    D2 = np.asarray(D2)
    D2 = np.array(D2, dtype = int)
    global gongShi # 工时=比例*订单加工总数
    gongShi=[]*B*M
    temp = 0
    for i in range(B):
        tem = [0]*M # 临时子工时
        tem[i] = np.multiply(X[0+temp:M+temp], t[i])
        gongShi.extend(tem[i])
        temp += M

    #print("优化后工时",gongShi)
    f1 = np.dot(gongShi, P[:B*M])
    f2 = np.dot(D2, d[:B*M])

    f = f1+f2*yunfei*100  # 计算目标函数值, 生产费用 = 工时*报价，运输费用=单位距离运费*距离决策矩阵*距离
    f = f/100
    #print('订单分配工时:',gongShi)
    #计算每个订单的成本
    global everyBcost
    everyBcost = [-1]*B
    temp = 0
    for i in range(B):
        everyBcost[i] = np.dot(gongShi[0+temp:M+temp], P[0+temp:M+temp]) + yunfei*np.dot(D2[0+temp:M+temp], d[0+temp:M+temp])*100
        temp += M
    # 计算每个供应商的成本
    global everyMcost
    everyMcost =  [-1]*B*M
    for i in range(B*M):
        everyMcost[i] = np.dot(gongShi[i], P[i]) + yunfei*np.dot(D2[i], d[i])*100
    #print('适应度',f)
    return f