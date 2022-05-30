# -*- coding: utf-8 -*-
# @Time     : 2021/4/18 1:35
# @Author   : ufy
# @Email    : antarm@outlook.com / 549147808@qq.com
# @file     : classify.py
# @info     : 本模块用于根据提取特征，对切片进行癌和非癌的分类,本模块支持二维数据的处理

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from typing import List
from ufy.core.metrics._classification import Evaluation


try:
    # 老版本
    from sklearn import joblib
except:
    # 新版本
    import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers
from xgboost import XGBClassifier
from tqdm import tqdm


def find_csv_in_dir(dirpath, allfiles=None):
    file_list = os.listdir(dirpath)
    for f in file_list:
        cur_f = os.path.join(dirpath, f)
        if os.path.isdir(cur_f):
            find_csv_in_dir(cur_f, allfiles)
        elif os.path.basename(f).split('.')[-1] == 'csv':
            allfiles.append(cur_f)


def load_data_csvs(filenames, target_cols=1, drop_id=True, id_cols=1, split_rate=0.7, shuffle=True):
    '''
    :param filename: csv数据文件名列表
    :param col_targets: 标签所在列的数量，可能是多标签,要求标签必须在最后几列
    :param drop_id: 是否删除id列
    :param id_endcols: id相关信息有多少列，默认前k列为id相关信息列
    :param split_rate: 换分比例，默认训练集占70%
    :return: 划分好的数据集(x_train,y_train),(x_test,y_test)
    '''
    data = []
    for filename in tqdm(filenames):
        print('load data:', filename)
        data_file = pd.read_csv(filename)
        data_file = np.array(data_file)
        if data == []:
            data = data_file
        else:
            data = np.vstack((data, data_file))
    if drop_id:
        data = data[:, id_cols:]
        data = data.astype('float32')

    if shuffle:
        # data = data[np.random.permutation(len(data))]
        np.random.seed(46)
        np.random.shuffle(data) 
    split_point = int(len(data) * split_rate)
    train_data = data[:split_point]
    test_data = data[split_point:]

    x_train, y_train = train_data[:, :-target_cols], train_data[:, -target_cols:]
    x_test, y_test = test_data[:, :-target_cols], test_data[:, -target_cols:]

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))


def plot_roc_curve(y_trues, y_pres, model_names=None, pos_label=1, threshhold=0.5, issave=False, savename='roc.jpg',
                   analysis_filename='results_analysis.txt'):
    '''
    :param y_trues: 真实标签
    :param y_pres:  预测值
    :param pos_label: 正例标签
    :param issave: 是否保存结果
    :param savename: 保存结果图片的名
    :return: 绘制ROC曲线
    '''
    if model_names == None:
        model_names = [str(i + 1) for i in range(len(y_pres))]
    fp = open(analysis_filename, 'w', encoding='utf-8')
    plt.figure(figsize=(20,16),dpi=300)
    for i in range(len(y_pres)):
        fpr, tpr, thresholds = sk.metrics.roc_curve(y_trues[i], y_pres[i], pos_label=pos_label)
        # print(fpr,'\n',tpr)
        roc_auc = sk.metrics.auc(fpr, tpr)
        # roc_auc = sk.metrics.roc_auc_score(y_trues[i], y_pres[i])
        print(model_names[i] + ':')
        y_pres_i = (y_pres[i] > threshhold).astype(int)
        print(sk.metrics.classification_report(y_trues[i], y_pres_i, digits=4))
        fp.write(model_names[i] + ':\n')
        fp.writelines(sk.metrics.classification_report(y_trues[i], y_pres_i, digits=4))
        fp.write('\n\n')
        plt.plot(fpr, tpr, label=model_names[i] + ' (AUC={0:.4f})'.format(roc_auc), lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    if issave:
        plt.savefig(savename)  # 要在show之前保存

    plt.show()


def min_max_normal(x):
    '''
    :param x: 是一个二维数据
    :return: 一个单位化的数据
    '''
    min_col = np.min(x, axis=0)
    max_col = np.max(x, axis=0)
    return (x - min_col) / (max_col - min_col + 1e-7)


def softmax_temp(x, T=1):
    '''
    :param x: 带处理的数据
    :param T: 温度，默认温度为1
    :return: 根据温度，来进行softmax变换，并返回变换结果
    '''
    x = np.array(x)
    x_row_max = np.max(x, axis=-1)
    # x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x / T)
    # x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    x_exp_row_sum = x_exp.sum(axis=-1)
    softmax = x_exp / x_exp_row_sum
    return softmax


def MLP_softmax(x_train, y_train, model_savepath='MLP_softmax.h5', istraining=False, max_epochs=10000,
                batch_size=32, validation_split=0.3, verbose=1,class_num=2,shape_in=778):
    if os.path.exists(model_savepath):
        model = tf.keras.models.load_model(model_savepath)
    else:
        istraining = True
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(shape_in,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(class_num, activation='softmax'))

        model.summary()

    if istraining:
        print(y_train.shape)
        y_train_onehot = tf.one_hot(y_train, depth=class_num)
        print(y_train_onehot.shape)
        y_train_onehot = tf.reshape(y_train_onehot, (-1, class_num))
        print(y_train_onehot.shape)

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        callback_list = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=8),
                         tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath, monitor='val_loss',
                                                            save_best_only=True)]
        model.fit(x_train, y_train_onehot, batch_size=batch_size, epochs=max_epochs, validation_split=validation_split,
                  verbose=verbose, callbacks=callback_list)

    return model


def MLP_sigmoid(x_train, y_train, model_savepath='MLP_softmax.h5', istraining=False, max_epochs=10000,
                batch_size=32, validation_split=0.3, verbose=1):
    if os.path.exists(model_savepath):
        model = tf.keras.models.load_model(model_savepath)
    else:
        istraining = True
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(778,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()

    if istraining:
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['acc'])
        callback_list = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=8),
                         tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath, monitor='val_loss',
                                                            save_best_only=True)]
        model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, validation_split=validation_split,
                  verbose=verbose, callbacks=callback_list)

    return model


def KNN_k_fold(x_train, y_train, model_savepath='knn.pkl', istraining=False):
    if os.path.exists(model_savepath):
        model = joblib.load(model_savepath)
    else:
        istraining = True
        model = sk.neighbors.KNeighborsClassifier()
    if istraining:
        
        joblib.dump(model, model_savepath)
    return model


def RandomForest(x_train, y_train, model_savepath='RandomForest.pkl', istraining=True):
    if os.path.exists(model_savepath):
        model = joblib.load(model_savepath)
    else:
        istraining = True
        model = RandomForestClassifier()
    if istraining:
        model.fit(x_train, y_train)
        joblib.dump(model, model_savepath)
    return model


def XGboost(x_train, y_train, model_savepath='xgboost.pkl', istraining=True):
    if os.path.exists(model_savepath):
        model = joblib.load(model_savepath)
    else:
        istraining = True
        model = XGBClassifier()
    if istraining:
        model.fit(x_train, y_train, verbose=True)
        # model.save_model(model_savepath)
        joblib.dump(model, model_savepath)
    return model


def model_test(x_test, x_test_normal, y_test, threshhold: float,models_path: List, models_name: List, savename: str,
               analysis_filename: str,
               summaryname: str):
    y_trues = [y_test for i in range(len(models_name))]
    y_preds = []
    for i in tqdm(range(len(models_path))):

        # input
        if 'MLP' in models_name[i] or 'normal' in models_name[i]:
            x = x_test_normal
        else:
            x = x_test

        # model
        if 'MLP' in models_name[i]:
            model = tf.keras.models.load_model(models_path[i])
        else:
            model = joblib.load(models_path[i])

        # predict
        if models_name[i] == 'MLP sigmoid':
            y_pred = model.predict_on_batch(x)[:, 0]
        elif models_name[i] == 'MLP softmax':
            y_pred = model.predict_on_batch(x)[:, 1]
        else:
            y_pred = model.predict_proba(x)[:, 1]
        y_preds.append(y_pred)

    plot_roc_curve(y_trues, y_preds, models_name, threshhold=threshhold, pos_label=1, issave=True, savename=savename,
                   analysis_filename=analysis_filename)

    summary_data = DataFrame(columns=['model', 'Accuracy', 'Kappa', 'sensitivity', 'Specificity', 'F1-score', 'AUC'])
    for i in range(len(models_name)):
        evaluate = Evaluation(model_name=models_name[i], y_true=y_trues[i], y_pred=y_preds[i], lables=[1, 0],thresh_hold=threshhold)
        summary_data.loc[i] = [evaluate.model_name, evaluate.accuracy, evaluate.kappa, evaluate.sensitivity,
                               evaluate.specificity, evaluate.f1_score, evaluate.auc]
    if not summaryname.endswith('.csv'):
        raise ValueError('请使用csv文件名！！！')
    summary_data.to_csv(summaryname)


if __name__ == '__main__':
    allfiles = []
    find_csv_in_dir('../data/diagnosis/new_cancer/', allfiles)
    print(allfiles)
