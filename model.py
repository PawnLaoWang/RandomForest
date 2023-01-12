import numpy as np
import random
import pandas as pd
from sklearn import ensemble
import time
import pickle


def train_maize(df, train_set):
    random.seed(11)
    clf = ensemble.RandomForestClassifier(n_estimators=100, min_samples_split=2, max_features=9, n_jobs=-1)  # 使用随机森林算法
    neg_inter = 5000  # 确保每个基因至少被选中一次的概率大于99%
    pickle.dump(neg_inter, pik_f)
    for i in range(0, neg_inter):
        train_data = train_set
        training_negative = random.sample(list(df[df['class'] == 0].index), int(len(train_data) * 10))  # 随机选择阴性基因
        train_data = train_data.append(df.iloc[training_negative])
        train_feature = train_data.drop(['class'], axis=1)
        clf.fit(train_feature, train_data['class'])
        pickle.dump(clf, pik_f)
    print('training complete')


if __name__ == '__main__':
    with open("model.dat", "wb") as pik_f:
        dt = './GO.csv'  # 输入文件
        start_time = time.time()
        df = pd.read_csv(dt)
        pickle.dump(df, pik_f)
        df = df.dropna(axis=1, how='all')
        df = df.drop(['ID'], axis=1)
        train_set = df[df['class'] == 1]
        train_maize(df, train_set)  # 进行训练

    print("--- %s seconds ---" % round((time.time() - start_time), 2))
