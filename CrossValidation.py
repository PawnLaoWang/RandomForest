import numpy as np
import random
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import sys
from scipy.interpolate import interp1d

import time

start_time = time.time()


def trains(df, train_set):
    k_numb_iter = 100
    clf = ensemble.RandomForestClassifier(n_estimators=100, min_samples_split=2,
                                          max_features=9)  # random forest parameters
    froc = open('GO' + "_" + "ROC.txt", 'w')  # output file for ploting ROC curve
    all_mean_tpr1 = [float()] * k_numb_iter
    ROC_all_auc = []
    importance = []
    if k_numb_iter > 1:
        shuffle_switch = True
    else:
        shuffle_switch = False
    neg_numb_iter = 50
    for n in range(0, k_numb_iter):
        skf = KFold(n_splits=5, shuffle=shuffle_switch)
        P_R_auc_mean = []
        ROC_auc_mean = []
        fold_mean_tpr1 = [0.0] * 101
        for m, (train_cv, test_cv) in enumerate(skf.split(train_set)):
            fold_mean_fpr = np.linspace(0, 1, 101)
            for i in range(0, neg_numb_iter):
                train_data = train_set.iloc[train_cv]
                test_data = train_set.iloc[test_cv]
                training_negative = random.sample(list(df[df['class'] == 0].index), int(len(train_data) * 10))
                testing_negatives = random.sample(list(df[df['class'] == 0].index), len(test_cv) * 200)
                train_data = train_data.append(df.iloc[training_negative])
                test_data = test_data.append(df.iloc[testing_negatives])
                train_feature = train_data.drop(['class'], axis=1)
                test_feature = test_data.drop(['class'], axis=1)
                probas_ = clf.fit(train_feature, train_data['class']).predict_proba(
                    test_feature)
                importance.append(clf.feature_importances_)
                prec, tpr, thresholds = precision_recall_curve(test_data['class'], probas_[:, 1])
                fpr, tpr1, thresholds1 = roc_curve(test_data['class'], probas_[:, 1])
                ROC_auc_mean.append(auc(fpr, tpr1))
                P_R_auc_mean.append(auc(tpr, prec))
                fun2 = interp1d(fpr, tpr1)
                fold_mean_tpr1 += fun2(fold_mean_fpr)

        fold_mean_tpr1 /= 5 * neg_numb_iter
        all_mean_tpr1[n] = fold_mean_tpr1
        ROC_all_auc.append(np.mean(ROC_auc_mean))

    all_mean_tpr1 = np.array(all_mean_tpr1)
    average_tpr1 = [float()] * 101
    sd_tpr1 = [float()] * 101
    for i in range(101):
        average_tpr1[i] = np.mean(all_mean_tpr1[:, i])
        sd_tpr1[i] = np.std(all_mean_tpr1[:, i])

    print('ROC AUC averge %f ; SD is %f' % (
    np.mean(ROC_all_auc), np.std(ROC_all_auc)))
    for i in range(0, len(fold_mean_fpr)):
        froc.write('{0}\t{1}\t{2}\n'.format(fold_mean_fpr[i], average_tpr1[i], sd_tpr1[i]))


dt = sys.argv[1]
df = pd.read_csv(dt)
df = df.drop(['ID'], axis=1)
df = df.dropna(axis=1, how='all')
train_set = df[df['class'] == 1]
trains(df, train_set)

print("--- %s seconds ---" % round((time.time() - start_time), 2))