# -*- coding: utf-8 -*-
"""
@brief : 文本特征提取
@author: Bayes
"""
import pandas as pd
import pickle
import time

"""=====================================================================================================================
2 特征工程
"""
def get_feature_tf(df_all, df_train, df_test, y_train, column_name, save=False):
    """1.1 原始文本数据--->tf特征"""
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8)
    vectorizer.fit(df_all[column_name])
    x_train = vectorizer.transform(df_train[column_name])
    x_test = vectorizer.transform(df_test[column_name])
    if save:
        data = (x_train, y_train, x_test)
        fp = open('./data_tf.pkl', 'wb')
        pickle.dump(data, fp)
        fp.close()
    return x_train,x_test
 
def get_feature_tfidf(df_all, df_train, df_test, y_train, column_name, save=False):
    """1.2 原始文本数据--->tfidf特征"""
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    vectorizer.fit(df_all[column_name])
    x_train = vectorizer.transform(df_train[column_name])
    x_test = vectorizer.transform(df_test[column_name])
    if save:
        data = (x_train, y_train, x_test)
        fp = open('./data_tfidf.pkl', 'wb')
        pickle.dump(data, fp)
        fp.close()
    return x_train,x_test

def get_feature_hash(df_all, df_train, df_test, y_train, column_name, save=False):
    """1.3 原始文本数据--->hash特征"""
    
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=200)
    df_all = vectorizer.fit_transform(df_all[column_name])
    x_train = df_all[:len(y_train)]
    x_test = df_all[len(y_train):]
    if save:
        data = (x_train.toarray(), y_train, x_test.toarray())
        fp = open('./data_hash.pkl', 'wb')
        pickle.dump(data, fp)
        fp.close()
    return x_train,x_test

def get_feature_doc2vec(df_all, df_train, df_test, y_train, column_name, save=False):
    """1.4 原始文本数据--->doc2vec特征"""
    
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    def sentence2list(sentence):
        s_list = sentence.strip().split()
        return s_list

    df_all[column_name] = df_all[column_name].apply(sentence2list)
    texts = df_all[column_name].tolist()

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=200, window=5, min_count=3, workers=4, epochs=25)
    docvecs = model.docvecs

    x_train = []
    for i in range(0, 102277):
        x_train.append(docvecs[i])
    x_train = np.array(x_train)

    x_test = []
    for j in range(102277, 204554):
        x_test.append(docvecs[j])
    x_test = np.array(x_test)

    if save:
        data = (x_train, y_train, x_test)
        fp = open('./data_doc2vec_25.pkl', 'wb')
        pickle.dump(data, fp)
        fp.close()
    return x_train,x_test
	
def get_feature_lda(tf_path, save=False):
    """2.1 tf特征降维--->lda特征"""
    
    from sklearn.decomposition import LatentDirichletAllocation
    
    """读取tf特征"""
    # tf_path = './feat_tf.pkl'
    f_tf = open(tf_path, 'rb')
    x_train, y_train, x_test = pickle.load(f_tf)
    f_tf.close()

    """特征降维：lda"""
    print("lda......")
    lda = LatentDirichletAllocation(n_components=200)
    x_train = lda.fit_transform(x_train)
    x_test = lda.transform(x_test)

    if save:
        data = (x_train, y_train, x_test)
        f_data = open('./feat_lda.pkl', 'wb')
        pickle.dump(data, f_data)
        f_data.close()
    return x_train,x_test

    
def get_feature_lsa(tfidf_path, save=False):
    """2.2 tfidf特征降维--->lsa特征"""
    from sklearn.decomposition import TruncatedSVD

    """读取tfidf特征"""
    # tfidf_path = './feat_tfidf.pkl'
    f_tfidf = open(tfidf_path, 'rb')
    x_train, y_train, x_test = pickle.load(f_tfidf)
    f_tfidf.close()

    """特征降维：lsa"""
    print("lsa......")
    lsa = TruncatedSVD(n_components=200)
    x_train = lsa.fit_transform(x_train)
    x_test = lsa.transform(x_test)

    if save:
        data = (x_train, y_train, x_test)
        f_data = open('./feat_lsa.pkl', 'wb')
        pickle.dump(data, f_data)
        f_data.close()
    return x_train,x_test
