# -*- coding:utf-8 -*-
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from defineArgs import *
import time
from sklearn import svm
import random
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def fisher(train_data):
    '''
    输入:样本数据集
    输出：特征重要程度排序
    '''
    rowNum, colNum = train_data.shape
    n = colNum - 1
    m = len(set(train_data[:, -1]))  # 由于采用二分类，这里m=2
    # outW = np.random.uniform(-1, 1, n)# 初始化权重矩阵
    outW = np.zeros(n)
    for i in range(n):
        icol_mean = train_data[:, i].mean()  # 计算第i列算术平均值
        temp1, temp2 = 0, 0
        for j in range(m):
            # 类间方差
            choose_v = train_data[:, -1] == j
            jdata = train_data[:, i][choose_v]  # 选出标签为j的i特征数据
            jmean, jvar = jdata.mean(), jdata.var(ddof=1)
            jcolvar = np.var([jmean, icol_mean], ddof=1)  # ddof代表计算方差除N-1
            temp1 += jcolvar
            temp2 += jvar
        if temp1 == 0:
            outW[i] = 0
        elif temp2 == 0:
            outW[i] = 100
        else:
            outW[i] = temp1 / temp2
    # 将特征按照重要程序降序排列
    decIdx = np.argsort(-outW)
    # decIdx=np.array([12,3,23,35,24,32,2,14,30,27,40,33,41,28,10,22,37,36,4,31,34,29,25,38,26,39,18,17,11,13,1,6,16,19,9,8,15,5,7,2,21])-1
    decOutW = outW[decIdx]
    # 将fisher运算结果存储到csv中
    # 将序号对应的名称存到列表
    nameList = list(map(lambda idx: num2featureDic[idx], decIdx))
    dataFrame = pd.DataFrame({'number': decIdx, 'name': nameList})
    dataFrame.to_csv(FISHER_FILE)
    # 按重要性程度输出特征名称，可选
    showStr = ' / '.join(
        map(lambda x: num2featureDic[x], decIdx))
    print("The priority order by: \n" + showStr)
    return decIdx


def KNNTest(select_train, train_label, select_test, test_label):
    # test kNN precision
    # K这个参数在defineArgs里面设置
    # 如果选取全部的trainset
    rightCnt, wrongCnt = 0, 0
    trainNum, testNum = select_train.shape[0], select_test.shape[0]
    resArr = []

    timeStart = time.time()
    for each in tqdm(select_test, total=testNum):
        this_res = kNNClassify(
            each, select_train, train_label, K)
        resArr.append(this_res)
    timeEnd = time.time()
    timeSpend = timeEnd - timeStart

    resArr = np.array(resArr)
    correctNum = (resArr == test_label).sum()
    wrongNum = testNum - correctNum

    correctRate, wrongRate = correctNum / testNum * 100, wrongNum / testNum * 100
    # 漏报,误报计算
    louBaoArr = ((resArr == 0).astype(np.int) +
                 (test_label > 0).astype(np.int)) == 2
    louBaoCnt = louBaoArr.sum()
    wuBaoCnt = wrongNum - louBaoCnt
    louBaoRate, wuBaoRate = louBaoCnt / testNum * 100, wuBaoCnt / testNum * 100
    print("KNN correctRate: %.2f , wrongRate: %.2f , louBaoRate: %.2f , wuBaoRate: %.2f , Time: %.2f s." %
          (correctRate, wrongRate, louBaoRate, wuBaoRate, timeSpend))


def kNNClassify(newInput, dataSet, labels, k):
    '''
    K近邻算法的实现
    '''
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = np.tile(newInput, (numSamples, 1)) - \
        dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = np.argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        # step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


def SVMtest(select_train, train_label, select_test, test_label):
    # 设置需要统计的参数
    rightCnt, wrongCnt = 0, 0
    trainNum, testNum = select_train.shape[0], select_test.shape[0]
    resArr = []
    # 构建svm模型并训练模型
    clf = svm.SVC(C=100)
    clf.fit(select_train, train_label)
    # 测试集验证
    timeStart = time.time()
    for each in tqdm(select_test, total=testNum):
        this_res = clf.predict(each.reshape(1, -1))
        resArr.append(this_res[0])
    timeEnd = time.time()
    timeSpend = timeEnd - timeStart

    resArr = np.array(resArr)
    correctNum = (resArr == test_label).sum()
    wrongNum = testNum - correctNum

    correctRate, wrongRate = correctNum / testNum * 100, wrongNum / testNum * 100
    # 漏报,误报计算
    louBaoArr = ((resArr == 0).astype(np.int) +
                 (test_label > 0).astype(np.int)) == 2
    louBaoCnt = louBaoArr.sum()
    wuBaoCnt = wrongNum - louBaoCnt
    louBaoRate, wuBaoRate = louBaoCnt / testNum * 100, wuBaoCnt / testNum * 100
    print("SVM correctRate: %.2f , wrongRate: %.2f , louBaoRate: %.2f , wuBaoRate: %.2f , Time: %.2f s." %
          (correctRate, wrongRate, louBaoRate, wuBaoRate, timeSpend))
    return wrongRate


def fruit_ag(argx, fun):
    '''

    :param argx: 需要优化的参数
    :param fun: 迭代函数
    :return: 最小误差、最优值
    '''
    # 果蝇算法
    # 初始化果蝇参数
    popsize = 30  # 果蝇种群规模
    maxgen = 15  # 果蝇最大迭代次数
    R = 0.01  # 果蝇飞行半径

    D = len(argx)  # 优化变量个数
    Dist = np.zeros([popsize, D])  # 记录位置
    S = np.zeros([popsize, D])  # 味道浓度判定值
    Smell = np.zeros([popsize, 1])  # 味道浓度值
    X = np.zeros([popsize, D])  # X轴位置信息
    Y = np.zeros([popsize, D])  # Y轴位置信息
    fitness = []  # 味道浓度路径
    S_path = []  # 味道判定值路径
    all_smell = [] # 全部比率信息

    # 赋予果蝇群体初始位置
    argx_ds = np.array([1 / x for x in argx])
    X_axis = np.sqrt(2) / 2 * argx_ds
    Y_axis = np.sqrt(2) / 2 * argx_ds

    # 赋予果蝇种群飞行半径
    for i in range(popsize):  # 遍历所有果蝇
        X[i, :] = X_axis + R * (2 * np.random.rand(1, D) - 1)
        Y[i, :] = Y_axis + R * (2 * np.random.rand(1, D) - 1)
        # 计算距离Dist
        Dist[i, :] = np.sqrt(X[i, :] ** 2 + Y[i, :] ** 2)
        # 计算味道浓度的倒数作为味道浓度判定值
        S[i, :] = 1 / Dist[i, :]
        # 带入味道浓度函数中求出味道浓度值
        Smell[i] = fun(S[i, 0], S[i, 1])[0]
    # 找出味道浓度最大值
    Smellbest, index = np.min(Smell), np.argmin(Smell)
    bestSmell = Smellbest
    S_path.append(S[index, :])
    # 保留最佳味道浓度处的果蝇
    X_axis = X[int(index), :]
    Y_axis = Y[int(index), :]
    # 果蝇种群进入迭代寻优
    for j in range(maxgen):  # while True:
        print('第{}次迭代'.format(j+1))
        smell_all = []
        #混沌变换
        # X_axis = hundun(X_axis)
        # Y_axis = hundun(Y_axis)
        for i in range(popsize):  # 遍历所有果蝇
            X[i] = X_axis + R * (2 * np.random.rand(1, D) - 1)
            Y[i] = Y_axis + R * (2 * np.random.rand(1, D) - 1)
            # 计算距离Dist
            Dist[i, :] = np.sqrt(X[i, :] ** 2 + Y[i, :] ** 2)
            # 计算味道浓度的倒数作为味道浓度判定值
            S[i, :] = 1 / Dist[i, :]
            # 带入味道浓度函数中求出味道浓度值
            smell_temp = fun(S[i, 0], S[i, 1]) # 修改
            Smell[i] = smell_temp[0]
            smell_all.append(smell_temp)

        Smellbest, index = np.min(Smell), np.argmin(Smell)
        all_smell.append(smell_all[index])
        print('smellbest:', Smellbest)
        if Smellbest < bestSmell:
            bestSmell = Smellbest
            X_axis = X[index, :]
            Y_axis = Y[index, :]
            S_path.append(S[index, :])
        fitness.append(bestSmell)
        print('path: ', S_path[-1])
        print('bestsmell: ', bestSmell)

    # 适应度曲线图
    all_smell1 = all_smell[np.argmin(all_smell[-1])]
    best_smell = min(fitness)
    best_index = fitness.index(best_smell)
    # plt.plot([x for x in range(len(fitness))], fitness, c='b')  # 画拟合数据曲线图
    # plt.scatter(best_index, best_smell, zorder=1, c="r")
    # plt.text(best_index, best_smell, "(%d, %.4f)" % (best_index, best_smell))
    # plt.title("Fitness and BestSmell")
    # plt.xlabel('iter')
    # plt.ylabel('MSE')
    # plt.show()
    print('correctRate: %.2f , wrongRate: %.2f , louBaoRate: %.2f , wuBaoRate: %.2f ' %
          ((100-best_smell), best_smell, all_smell1[2], all_smell1[3]))
    return best_smell, S_path[-1], fitness, best_index


def hundun(x):
    '''

    :param x: 混沌变换前的值
    :return: 混沌变换后的值
    '''
    # 设置搜寻边界
    x_lb = 0.01
    x_ub = 150
    D = 2
    if (x < x_lb).any() or (x > x_ub).any():  # 判断是否超出边界
        print('超出边界')
        x_1 = np.random.rand(1, D) # 如果出界，则随机取值
        # 如果超出边界，则更新边界
        # range_lb_ub = ub - lb
        # lb = x - range_lb_ub * x_1
        # ub = x + range_lb_ub * (1 - x_1)
    else:  # 如果没有超出边界
        # 归一化
        x_1 = (x - x_lb) / (x_ub - x_lb)
    # 混沌变换
    x_2 = 4 * x_1 * (1 - x_1)
    # 反归一化
    x_3 = x_lb + x_2 * (x_ub - x_lb)
    return x_3


def SVM_FOA(C, gamma):
    '''

    :param C:
    :param gamma:
    :return: 需要优化的错误率
    '''
    trainNum, testNum = select_train.shape[0], select_test.shape[0]
    resArr = []
    # 构建svm模型并训练模型
    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    clf.fit(select_train, train_label)
    # 测试集验证
    timeStart = time.time()
    for each in tqdm(select_test, total=testNum):
        this_res = clf.predict(each.reshape(1, -1))
        resArr.append(this_res[0])
    timeEnd = time.time()
    timeSpend = timeEnd - timeStart

    resArr = np.array(resArr)
    correctNum = (resArr == test_label).sum()
    wrongNum = testNum - correctNum

    correctRate, wrongRate = correctNum / testNum * 100, wrongNum / testNum * 100
    # 漏报,误报计算
    louBaoArr = ((resArr == 0).astype(np.int) +
                 (test_label > 0).astype(np.int)) == 2
    louBaoCnt = louBaoArr.sum()
    wuBaoCnt = wrongNum - louBaoCnt
    louBaoRate, wuBaoRate = louBaoCnt / testNum * 100, wuBaoCnt / testNum * 100
    print("SVM correctRate: %.2f , wrongRate: %.2f , louBaoRate: %.2f , wuBaoRate: %.2f , Time: %.2f s." %
          (correctRate, wrongRate, louBaoRate, wuBaoRate, timeSpend))
    return wrongRate, correctRate, louBaoRate, wuBaoRate


if __name__ == '__main__':
    start_time = time.time()
    # 加载训练集
    train_data = pickle.load(open(TRAINDATA, 'rb'))

    # # 如果没有FISHER_FILE,需要使用fisher选取特征，如果存在，就直接读从FISHER_FILE中读取
    # if os.path.exists(FISHER_FILE):
    #     decFeatureIdx = np.array(pd.read_csv(FISHER_FILE)['number'])
    # else:
    #     decFeatureIdx = fisher(train_data)
    decFeatureIdx = fisher(train_data)

    # 采样训练集
    train_data_num = train_data.shape[0]
    selectIdx_list = random.sample(
        range(train_data_num), int(train_data_num * SAMPLE_TRAIN_RATE))
    train_data = train_data[selectIdx_list, :]

    select_train = train_data[:, decFeatureIdx[:YOU_CHOOSE]]
    # PCA
    # pca = PCA(n_components=10)
    # select_train = pca.fit_transform(select_train)

    train_label = train_data[:, -1].astype(np.int)
    # -----------------
    # 加载测试集
    test_data = pickle.load(open(TESTDATA, 'rb'))
    # 采样测试集
    test_data_num = test_data.shape[0]
    selectIdx_list = random.sample(
        range(test_data_num), int(test_data_num * SAMPLE_TEST_RATE))
    test_data = test_data[selectIdx_list, :]
    # 根据feature选择测试集的相关列
    # select_test = test_data[:, decFeatureIdx[[2]]]
    select_test = test_data[:, decFeatureIdx[:YOU_CHOOSE]]
    # PCA
    # select_test = pca.fit_transform(select_test)

    test_label = test_data[:, -1].astype(np.int)
    print("Train data num : %d , Test data num : %d" %
          (select_train.shape[0], select_test.shape[0]))

    # 单独SVM分类
    print('单独SVM：')
    w_rate = SVMtest(select_train, train_label, select_test, test_label)
    print("----SVM-FOA Test----")

    # foa算法优化及其可视化：
    # 初始化参数，相当于给果蝇寻优设置起始点
    C = 100
    gamma = 8
    mse, args, fitness, best_index = fruit_ag([C, gamma], SVM_FOA)
    x = [0] + [x+1 for x in range(len(fitness))]
    y = [w_rate] + fitness
    plt.plot(x, y, c='b')  # 画拟合数据曲线图
    plt.scatter(best_index+1, min(fitness), zorder=1, c="r")
    plt.text(best_index, min(fitness), "(%d, %.4f)" % (best_index+1, min(fitness)))
    plt.title("Fitness and BestSmell")
    plt.xlabel('iter')
    plt.ylabel('MSE')
    plt.show()

    print('优化过后错误率:', mse)
    print('优化后的[C  gamma]', args)
    end_time = time.time()
    print('time cost:', end_time-start_time)

    print('运用foa_svm后，提升 %.2f 个百分点' % (w_rate - mse))