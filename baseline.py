"""
@简介：Baseline (常规的处理流程)
"""
#导入所需要的软件包
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

print("开始...............")

#====================================================================================================================
# @代码功能简介：读取数据，并进行简单处理
# @知识点定位：数据预处理
#====================================================================================================================
df_train = pd.read_csv('../data/train_set.csv')
df_test = pd.read_csv('../data/test_set.csv')

#====================================================================================================================
# @代码功能简介：提取数据中的数值特征，若有文本，提取文本特征
# @知识点定位：特征工程
#====================================================================================================================
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
vectorizer.fit(df_train['text'])
x_train = vectorizer.transform(df_train['text'])
x_test = vectorizer.transform(df_test['text'])
y_train = df_train['label']

#====================================================================================================================
# @代码功能简介：选取合适的分类器，并进行训练
# @知识点定位：算法模型
#=====================================================================================================================
clf = LinearSVC()
clf.fit(x_train, y_train)

#====================================================================================================================
# @代码功能简介：测试集预测
# @知识点定位：评价模型
#====================================================================================================================
y_test = clf.predict(x_test)

#====================================================================================================================
# @代码功能简介：将测试集的预测结果保存至本地
# @知识点定位：输出结果
#====================================================================================================================
df_test['label'] = y_test.tolist()
df_result = df_test.loc[:, ['id', 'label']]
df_result.to_csv('../results/baseline.csv', index=False)

print("完成...............")
