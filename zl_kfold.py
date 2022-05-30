from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from ufy.core.metrics._classification import Evaluation
from sklearn.model_selection import cross_val_score
import sklearn as sk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers
from ufy.HX.diagnosis.classify import joblib
import keras
from sklearn.model_selection import GridSearchCV

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

def kfold_knn(x_train, y_train, threshhold=0.5,num_folds = 5,summaryname = 'table.csv'):
    acc_per_fold = []
    loss_per_fold = []
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    auc_scores = []
    summary_data = DataFrame(columns=['model', 'Accuracy', 'Kappa', 'sensitivity', 'Specificity', 'F1-score', 'AUC'])
    knn = sk.neighbors.KNeighborsClassifier()
    for train, test in kfold.split(x_train, y_train):
        x = x_train[test]
        knn.fit(x_train[train],y_train[train])
        y_preds = y_pred = knn.predict_proba(x)[:, 1]
        evaluate = Evaluation(model_name='knn'+str(fold_no), y_true=y_train[test], y_pred=y_preds, lables=[1, 0],
                              thresh_hold=threshhold)
        summary_data.loc[fold_no] = [evaluate.model_name, evaluate.accuracy, evaluate.kappa, evaluate.sensitivity,
                               evaluate.specificity, evaluate.f1_score, evaluate.auc]
        fold_no = fold_no+1
    summary_data.loc[fold_no] = ['平均值', summary_data['Accuracy'].mean(), summary_data['Kappa'].mean(),
                                 summary_data['sensitivity'].mean(), summary_data['Specificity'].mean(),
                                  summary_data['F1-score'].mean(), summary_data['AUC'].mean()]
    summary_data.to_csv(summaryname)

    
def kfold_Random_Forest(x_train, y_train, threshhold=0.5,num_folds = 5,summaryname = 'table.csv'):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    summary_data = DataFrame(columns=['model', 'Accuracy', 'Kappa', 'sensitivity', 'Specificity', 'F1-score', 'AUC'])
    model = RandomForestClassifier()
    for train, test in kfold.split(x_train, y_train):
        x = x_train[test]
        model.fit(x_train[train], y_train[train])
        y_preds = model.predict_proba(x)[:, 1]
        evaluate = Evaluation(model_name='Random_Forest'+str(fold_no), y_true=y_train[test], y_pred=y_preds, lables=[1, 0],
                              thresh_hold=threshhold)
        summary_data.loc[fold_no] = [evaluate.model_name, evaluate.accuracy, evaluate.kappa, evaluate.sensitivity,
                               evaluate.specificity, evaluate.f1_score, evaluate.auc]
        fold_no = fold_no+1
    summary_data.loc[fold_no] = ['平均值', summary_data['Accuracy'].mean(), summary_data['Kappa'].mean(),
                                 summary_data['sensitivity'].mean(), summary_data['Specificity'].mean(),
                                  summary_data['F1-score'].mean(), summary_data['AUC'].mean()]
    summary_data.to_csv(summaryname)
    
    
def kfold_Xgboost(x_train, y_train, threshhold=0.5,num_folds = 5,summaryname = 'table.csv'):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    summary_data = DataFrame(columns=['model', 'Accuracy', 'Kappa', 'sensitivity', 'Specificity', 'F1-score', 'AUC'])
    model = XGBClassifier()
    for train, test in kfold.split(x_train, y_train):
        x = x_train[test]
        model.fit(x_train[train], y_train[train])
        y_preds = model.predict_proba(x)[:, 1]
        evaluate = Evaluation(model_name='Xgboost'+str(fold_no), y_true=y_train[test], y_pred=y_preds, lables=[1, 0],
                              thresh_hold=threshhold)
        summary_data.loc[fold_no] = [evaluate.model_name, evaluate.accuracy, evaluate.kappa, evaluate.sensitivity,
                               evaluate.specificity, evaluate.f1_score, evaluate.auc]
        fold_no = fold_no+1
    summary_data.loc[fold_no] = ['平均值', summary_data['Accuracy'].mean(), summary_data['Kappa'].mean(),
                                 summary_data['sensitivity'].mean(), summary_data['Specificity'].mean(),
                                  summary_data['F1-score'].mean(), summary_data['AUC'].mean()]
    summary_data.to_csv(summaryname)
    




def kfold_MLP_softmax(x_train, y_train, threshhold=0.5,num_folds = 5,summaryname = 'table.csv', model_savepath='MLP_softmax',
                      max_epochs=10000, batch_size=32, validation_split=0.3, verbose=1):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    summary_data = DataFrame(columns=['model', 'Accuracy', 'Kappa', 'sensitivity', 'Specificity', 'F1-score', 'AUC'])
    for train, test in kfold.split(x_train, y_train):
        x = x_train[test]
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(884,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(2, activation='softmax'))

        model.summary()
        print(y_train.shape)
        y_train_onehot = tf.one_hot(y_train[train], depth=2)
        print(y_train_onehot.shape)
        y_train_onehot = tf.reshape(y_train_onehot, (-1, 2))
        print(y_train_onehot.shape)

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        callback_list = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=8),
                         tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath+str(fold_no)+'.h5', monitor='val_loss',
                                                            save_best_only=True)]

        model.fit(x_train[train], y_train_onehot, batch_size=batch_size, epochs=max_epochs, validation_split=validation_split,
                  verbose=verbose, callbacks=callback_list)
        y_preds = model.predict_proba(x)[:, 1]
        evaluate = Evaluation(model_name='MLP_softmax'+str(fold_no), y_true=y_train[test], y_pred=y_preds, lables=[1, 0],
                              thresh_hold=threshhold)
        summary_data.loc[fold_no] = [evaluate.model_name, evaluate.accuracy, evaluate.kappa, evaluate.sensitivity,
                               evaluate.specificity, evaluate.f1_score, evaluate.auc]
        fold_no = fold_no+1
    summary_data.loc[fold_no] = ['平均值', summary_data['Accuracy'].mean(), summary_data['Kappa'].mean(),
                                 summary_data['sensitivity'].mean(), summary_data['Specificity'].mean(),
                                  summary_data['F1-score'].mean(), summary_data['AUC'].mean()]
    summary_data.to_csv(summaryname)
    
    
    
def kfold_MLP_sigmoid(x_train, y_train, threshhold=0.5,num_folds = 5,summaryname = 'table.csv', model_savepath='MLP_softmax',
                      max_epochs=10000, batch_size=32, validation_split=0.3, verbose=1):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    summary_data = DataFrame(columns=['model', 'Accuracy', 'Kappa', 'sensitivity', 'Specificity', 'F1-score', 'AUC'])
    for train, test in kfold.split(x_train, y_train):
        x = x_train[test]
        istraining = True
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(884,)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['acc'])
        callback_list = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=8),
                         tf.keras.callbacks.ModelCheckpoint(filepath=model_savepath+str(fold_no)+'.h5', monitor='val_loss',
                                                            save_best_only=True)]
        model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, validation_split=validation_split,
                  verbose=verbose, callbacks=callback_list)
        y_preds = model.predict_proba(x)[:, 0]
        evaluate = Evaluation(model_name='MLP_sigmoid'+str(fold_no), y_true=y_train[test], y_pred=y_preds, lables=[1, 0],
                              thresh_hold=threshhold)
        summary_data.loc[fold_no] = [evaluate.model_name, evaluate.accuracy, evaluate.kappa, evaluate.sensitivity,
                               evaluate.specificity, evaluate.f1_score, evaluate.auc]
        fold_no = fold_no+1
    summary_data.loc[fold_no] = ['平均值', summary_data['Accuracy'].mean(), summary_data['Kappa'].mean(),
                                 summary_data['sensitivity'].mean(), summary_data['Specificity'].mean(),
                                  summary_data['F1-score'].mean(), summary_data['AUC'].mean()]
    summary_data.to_csv(summaryname)
    
    
def KNN_gridserch(x_train, y_train):
    knn = sk.neighbors.KNeighborsClassifier()
    param_grid = [
        {
            'weights': ["uniform"],
            'n_neighbors': [i for i in range(1, 11)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]
    grid_search = GridSearchCV(knn, param_grid, scoring="roc_auc", cv=3, n_jobs=-1 ,verbose = 3)
    grid_search.fit(x_train,y_train)
    print('knn最佳auc：' + str(grid_search.best_score_))
    return grid_search