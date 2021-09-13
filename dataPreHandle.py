# -*- coding:utf-8 -*-
import numpy as np
import csv
from tqdm import tqdm
import pandas as pd
import pickle
from defineArgs import *

#------------step1对train和test数据做预处理------------
print('\n==========================Data Prehandle============================\n')
for SOURCE, SAVE in zip([RAW_TRAIN_PERCENT10, RAW_TEST], [PRE_HANDLE_TRAIN, PRE_HANDLE_TEST]):
    print('%s processing.....' % SOURCE)
    # 取得文件行数，方便tqdm显示
    NUMLINES = 0
    with open(SOURCE) as f:
        for line in f:
            if line.strip():
                NUMLINES += 1
    with open(SOURCE) as f:
        o = open(SAVE, 'w')
#         num=len(f)

        for line in tqdm(f, total=NUMLINES):
            try:
                l = line.strip().split(',')
                # 验证l是42维数据
                assert len(l) == 42
                # 字符型标签数值化
                l[1], l[2], l[3] = proto2numDic[l[1]
                                                ], service2numDic[l[2]], flag2numDic[l[3]]
                # label数值化（二值化,0代表normal,1代表unnormal）
                label = l[-1]
                l[-1] = 0 if label == NORMFLAG else 1
                handled_line = ','.join(map(lambda i: str(i), l)) + '\n'
                o.write(handled_line)
            except:
                pass  # 报错说明存在异常数据，pass掉
        o.close()

#------------step1对train和test数据做标准化和归一化------------
# 离散数据的列数：1,2,3,6,11,13,14,20,21,剩下的都是连续数据
# 另外需要考虑第19列数据 (0-index) : num_out_bound_cmds,全部都是0，当做离散数据处理
# 只对连续数据做标准化和归一化
print('\n================Data Standardization&Normalization==================\n')
discrete_colIndex = [1, 2, 3, 6, 11, 13, 14, 20, 21] + [19]
for SOURCE, SAVE_CSV, SAVE_PICKLE in zip([PRE_HANDLE_TRAIN, PRE_HANDLE_TEST], [NORM_TRAIN, NORM_TEST], [TRAINDATA, TESTDATA]):
    df = pd.read_csv(SOURCE, header=None, sep=',')
    print('%s processing.....' % SOURCE)
    for colIdx in range(0, 41):
        # 离散数据不处理
        if colIdx in discrete_colIndex:
            continue
        # 取出当前列的Series数据
        thisSeries = df[colIdx]
        # 标准化
        mean, std = thisSeries.mean(), thisSeries.std()
        thisSeries = (thisSeries - mean) / std
        # 归一化
        minV, maxV = thisSeries.min(), thisSeries.max()
        thisSeries = (thisSeries - minV) / (maxV - minV)
        # 赋值给原表数据
        df[colIdx] = thisSeries
    # 存储为csv文件，不保留行索引和列索引,方便观查数据样式
    print('     - 1.saving to csv file %s...' % (SAVE_CSV))
    df.to_csv(SAVE_CSV, header=None, index=0)
    # 另外存储为pickle二进制文件，方便后续任务读取
    print('     - 2.saving to pickle file %s...' % (SAVE_PICKLE))
    pickle.dump(np.array(df), open(SAVE_PICKLE, 'wb'))


print('\nData PreProcess Done! \n')
