# -*- coding: utf-8 -*-
"""
@brief : 通过GridSearchCV自动搜索分类器的最优的超参数值
"""
from sklearn.model_selection import GridSearchCV

def find_params(x_train, y_train, clf):

    print("start finding the best_params......")
    params = {'penalty':['l2', 'l1'], 'C':[1.0, 2.0, 3.0]}

    # GridSearchCV
    param1 = {'max_depth': [3, 5, 7],
              'min_child_weight': [2, 4]}
    param2 = {'learning_rate': [0.02, 0.05],
               'n_estimators': [200, 500, 800], }
    param3 = {'subsample': [0.8, 0.9, 1.0],
              'colsample_bytree': [0.8, 0.9, 1.0]}
    param4 = {'reg_alpha': [0.1, 1],
              'reg_lambda': [0.1, 1]}

    gsearch = GridSearchCV(estimator=clf, param_grid=param1, scoring='f1_macro', cv=5,
                           n_jobs=-1)  # 分类器默认为StratifiedKfold
    gsearch.fit(x_train, y_train)
    print('The best_params is:', gsearch.best_params_, 'The score is %.5g' % gsearch.best_score_)
    
    return gsearch
