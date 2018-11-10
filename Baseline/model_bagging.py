# -*- coding: utf-8 -*-
"""
@brief : 单模型k折bagging。(用K个模型分别对测试集进行预测，并得到K个结果，再进行结果投票或取均值)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time
import pickle

def model_bagging(x_train, y_train, x_test, clf):
    print("开始......")
    t_start = time.time()

    preds = []
    i = 0
    skf = StratifiedKFold(n_splits=10, random_state=1)
    score_sum = 0
    for train_idx, vali_idx in skf.split(x_train, y_train):
        i += 1
        """获取训练集和验证集"""
        f_train_x = x_train[train_idx]
        f_train_y = y_train[train_idx]
        f_vali_x = x_train[vali_idx]
        f_vali_y = y_train[vali_idx]

        """训练分类器"""
        clf.fit(f_train_x, f_train_y)

        """对测试集进行预测"""
        y_test = clf.predict(x_test)
        preds.append(y_test)

        """对验证集进行预测，并计算f1分数"""
        pre_vali = clf.predict(f_vali_x)
        score_vali = f1_score(y_true=f_vali_y, y_pred=pre_vali, average='macro')
        print("第{}折， 验证集分数：{}".format(i, score_vali))
        score_sum += score_vali
        score_mean = score_sum / i
        print("第{}折后， 验证集分平均分数：{}".format(i, score_mean))

    """对K个模型的结果进行融合，融合策略:投票机制"""
    preds_arr = np.array(preds).T
    y_test = []
    for pred in preds_arr:
        result_vote = np.argmax(np.bincount(pred))
        y_test.append(result_vote)

    t_end = time.time()
    print("训练结束，耗时:{}min".format((t_end - t_start) / 60))

    return y_test
