import numpy as np
import random
import pandas as pd
import time
import itertools as it
import argparse
import pickle


def train_prediction(df, Validation_set_i):
    prediction_list = (len(Validation_set_i)) * [0]
    validation_feature_all = Validation_set_i.drop(['class'], axis=1)
    if len(validation_feature_all) > 0:
        neg_inter = pickle.load(pik_f)
        print(neg_inter)
        for i in range(0, neg_inter):
            single_model = pickle.load(pik_f)
            validation_pred = single_model.predict_proba(validation_feature_all)[:, 1]
            prediction_list += validation_pred
        prediction_list_all = np.array(prediction_list)
        Ranks = prediction_list_all.argsort()[
                ::-1].argsort()
        prediction_list_freq = [l / neg_inter for l in
                                prediction_list]
        rank_list_df = pd.DataFrame({'ID': Validation_set_ID_uni, 'freq': prediction_list_freq})
        for i in gene_ex:
            rank_list_df = rank_list_df.append({'ID': i, 'freq': 0}, ignore_index=True)
        rank_list_df['Rank'] = rank_list_df['freq'].rank(ascending=0,
                                                         method='average')
        rank_list_df_sorted = rank_list_df.sort_values(by=['Rank'], ascending=True)
        rank_list_df_sorted = rank_list_df_sorted.reset_index(drop=True)
    else:
        if len(gene_ex) != 0:
            rank_list_df_sorted = pd.DataFrame({'ID': [], 'freq': [], 'Rank': []})
            for i in gene_ex:
                rank_list_df_sorted = rank_list_df_sorted.append({'ID': i, 'freq': 'NA', 'Rank': 'NA'},
                                                                 ignore_index=True)

    with open('GO_gene_rank.csv', 'a') as indenti_f:
        indenti_f.write('//' + '\n' + Gene_name + '\n')
        indenti_f.write('ID' + ',' + 'Rank_in_a_GO' + ',' + 'Frequency' + '\n')
        for i in range(len(rank_list_df_sorted)):
            indenti_f.write(rank_list_df_sorted['ID'][i] + ',' + str(rank_list_df_sorted['Rank'][i]) + ',' + str(
                rank_list_df_sorted['freq'][i]) + '\n')


if __name__ == '__main__':
    start_time = time.time()
    with open('./GOrank.scv', 'r') as f:
        for key, group in it.groupby(f, lambda line: line.startswith('//')):
            if not key:
                with open("./model_training/model.dat", "rb") as pik_f:
                    df = pickle.load(pik_f)
                    group = list(group)
                    group = [i.strip('\n') for i in group]
                    Gene_name = group[0]
                    print('GO: ' + Gene_name)
                    genes_in_GO = group[1:]
                    original_length = len(genes_in_GO)
                    Validation_set = pd.DataFrame()
                    for i in range(len(genes_in_GO)):
                        Validation_set = Validation_set.append(
                            df[df.ID == genes_in_GO[i]])

                    df = df.drop(['ID'], axis=1)
                    ind_for_exclusion = []
                    for t in range(
                            len(Validation_set.index)):
                        if Validation_set['class'][Validation_set.index[t]] == 1:
                            ind_for_exclusion.append(t)
                            print('Known GO gene excluded: ')
                            print(Validation_set['ID'][Validation_set.index[t]])
                    Validation_set = Validation_set.drop(
                        Validation_set.index[ind_for_exclusion])
                    Validation_set_ID_uni = list(Validation_set.ID)
                    print('Number of genes: ' + str(original_length))
                    gene_ex = (set(genes_in_GO) - set(
                        Validation_set_ID_uni))
                    Validation_set = Validation_set.drop(['ID'], axis=1)
                    Validation_set_i = Validation_set.reset_index(drop=True)
                    train_prediction(df, Validation_set_i)
    print("--- %s seconds ---" % round((time.time() - start_time), 2))
