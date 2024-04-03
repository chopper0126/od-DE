import numpy as np


def init_data(O,M,Oir):
    mean_v = 90
    yunfei = 5  # 单位距离配送价格
    amin = [0] * M * Oir * O
    # 初始化数据
    d = np.random.randint(100,300,(M))
    P = np.random.randint(100,300,(M*Oir))
    amax = np.random.randint(10,300,(M*Oir))*100
    good_p = np.random.randint(0, 100, (M*Oir*O))
    EC = np.random.randint(5, 100, (M*Oir)) # 供应商对于不同资源的能耗  功率
    E = np.random.randint(4000, 5000, (O))*100 # 供应商对于不同资源的能耗  功率
    Co = np.random.randint(5, 10, (O))# 供应商对于不同资源的能耗  功率
    t = np.random.randint(400/Oir, 600/Oir, (O*Oir))# 供应商对于不同资源的能耗  功率
    Pir = np.random.randint(1, 3, (O*Oir))# 供应商对于不同资源的能耗  功率
    print('d =',d.tolist()*O*Oir)
    print("P =",P.tolist()*O)
    print("amax =",amax.tolist())
    print("good_p=",good_p.tolist())
    print("EC=",EC.tolist())
    print("E =",E.tolist())
    print("Co =",Co.tolist())
    print("t =",t.tolist())
    print("Pir =",Pir.tolist())
    print("-"*10)
  # 供应商对于不同资源的能耗  功率
    return  d.tolist()*O*Oir,P.tolist()*O, t.tolist(), yunfei, amin, amax.tolist() ,Co.tolist() ,Pir.tolist() ,mean_v,good_p,E.tolist(),EC.tolist()
