from data_collection.reddit_user_dataset import RedditUserDataset
from classification.feature_computing import SavedPostBertEmbedder
from classification.classification_baseline import perform_training_user_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import datetime
import pandas as pd
import pickle as pickle
import itertools
import datetime
import pickle as pkl
import datetime
import os
import re
import numpy as np
from utils.train_utils import get_embeddings_dict_from_path
from utils.train_utils import write_embeddings
import gzip
from random import randrange


def divide_date(date1, date2, intervals):
    dates = [date1]
    delta = (date2-date1).total_seconds()/intervals
    for i in range(0, intervals):
      dates.append(dates[i] + datetime.timedelta(0,delta))
    return dates

def monthly_baseline():
    clf1 = LinearSVC(max_iter=100000)
    data_path = "data/core_dataset/core_dataset.gzip"
    #for i in range(1, 12):
    #    print(i)
    #    time_split = [(datetime.date(2020, i, 1), datetime.date(2020, i+1, 1)),  (datetime.date(2021, 2, 1), datetime.date(2021, 3, 1))
    #        , (datetime.date(2021, 3, 1), datetime.date(2021, 4, 1))]
    #    perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='bert', run_id='monthly_baseline')
    time_split = [(datetime.date(2020, 12, 1), datetime.date(2021, 1, 1)),  (datetime.date(2021, 2, 1), datetime.date(2021,3,1))
            , (datetime.date(2021, 3, 1), datetime.date(2021, 4, 1))]
    perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='bert', run_id='monthly_baseline')
    time_split = [(datetime.date(2021, 1, 1), datetime.date(2021, 2, 1)),  (datetime.date(2021, 2, 1), datetime.date(2021,3,1))
            , (datetime.date(2021, 3, 1), datetime.date(2021, 4, 1))]
    perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='bert', run_id='monthly_baseline')

def baselines_simple():
    clf1 = LinearSVC(max_iter=100000)
    data_path = "data/core_dataset/reddit_corpus_final_balanced.gzip"
    split = (datetime.date(2020, 1, 1), datetime.date(2021, 4, 25))
    #listdates = divide_date(split[0], split[0], 15)
    
    
    #time_split = [(datetime.date(2020, 11, 1), datetime.date(2021, 4, 1)),  (datetime.date(2020, 6, 1), datetime.date(2020,11,1))
    #        , (datetime.date(2020, 1, 1), datetime.date(2020, 6, 1))]
    #time_split = [(datetime.date(2020, 1, 1), datetime.date(2020, 6, 1)),  (datetime.date(2020, 6, 1), datetime.date(2020,11,1))
    #        , (datetime.date(2020, 11, 1), datetime.date(2021, 4, 1))]
    time_split = [(datetime.date(2020, 1, 1), datetime.date(2020, 8, 28)),  (datetime.date(2020, 8, 28), datetime.date(2020,12,26))
            , (datetime.date(2020, 12, 26), datetime.date(2021, 4, 25))]
    #perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='ngrams', mode='porter_stemmer' ,run_id='ngram_timesplit_porter')
    perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=0, folds=4,
                                feature_calc='bert', mode='None', run_id="usersplit_update")

    #perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='ngrams', mode='wordnet_lemmatizer' ,run_id='ngram_timesplit_lemma')
    #perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=0, folds=4,
    #                            feature_calc='ngrams', mode='wordnet_lemmatizer', run_id="ngram_usersplit_lemma")
    #perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=1, folds=4,
    #                            feature_calc='ngrams', run_id="ngram_usersplit")
    #perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=2, folds=4,
    #                            feature_calc='ngrams', run_id="ngram_usersplit")
    #perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=3, folds=4,
    #                            feature_calc='ngrams', run_id="ngram_usersplit")

def baselines():
    clf1 = LinearSVC(max_iter=100000)
    data_path = "data/core_dataset/reddit_corpus_final_balanced.gzip"
    split = (datetime.date(2020, 1, 1), datetime.date(2021, 4, 1))
    listdates = divide_date(split[0], split[1], 15)

    time_split = [ (datetime.date(2020, 1, 1), datetime.date(2020, 6, 1)),  (datetime.date(2020, 6, 1), datetime.date(2020,11,1))
            , (datetime.date(2020, 11, 1), datetime.date(2021, 4, 1))]
    perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=0, folds=4,
                                feature_calc='bert', run_id="user_split_baseline")
    perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=1, folds=4,
                                feature_calc='bert', run_id="user_split_baseline")
    perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=2, folds=4,
    feature_calc='bert', run_id="user_split_baseline")
    perform_training_user_split(data_path, split, clf1, split_type='user', fold_index=3, folds=4,
    feature_calc='bert', run_id="user_split_baseline")
    
    perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='bert', run_id='time_split_baseline')

    time_split_15 = []
    for tp in range(len(listdates)-1):
        time_split_15.append((listdates[tp], listdates[tp+1])) 
    
    cnt = 0
    for ts in time_split_15[:-2]:
        time_split = (ts, time_split_15[-2], time_split_15[-1])
        run_id='time_split_baseline_15_'+str(cnt)
        print(time_split)
        perform_training_user_split(data_path, time_split, clf1, split_type='time', fold_index=-1, feature_calc='bert', run_id=run_id)
        cnt+=1

    
def count_links(post):
    links = re.findall(
        r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
        post)
    return len(links)


def link_statistic():
    source_frame = pd.read_pickle('data/core_dataset/core_dataset.gzip', compression='gzip')
    spreader_links = []
    rn_links = []
    for index, row in source_frame.iterrows():
        val = 0
        for doc in row['documents']:
            val += count_links(doc[1])
        if row['fake_news_spreader'] == 1:
            spreader_links.append(val)
        elif row['fake_news_spreader'] == 0:
            rn_links.append(val)
        else:
            raise Exception("Invalid data!")

    print("Spreader:")
    print(np.median(spreader_links))

    print("Real News:")
    print(np.median(rn_links))


def subreddit_statistic():
    source_frame = pd.read_pickle('data/core_dataset/reddit_corpus_final_balanced.gzip', compression='gzip')
    all_sr = set()
    spreader_subreddits = {}
    rn_subreddits = {}
    for index, row in source_frame.iterrows():
        seen = []
        if row['fake_news_spreader'] == 1:
            for doc in row['documents']:
                all_sr.add(doc[3])
                if doc[3] in seen:
                    continue
                if doc[3] in spreader_subreddits.keys():
                    spreader_subreddits[doc[3]] += 1
                else:
                    spreader_subreddits[doc[3]] = 1
                seen.append(doc[3])
        elif row['fake_news_spreader'] == 0:
            for doc in row['documents']:
                all_sr.add(doc[3])
                if doc[3] in seen:
                    continue
                if doc[3] in rn_subreddits.keys():
                    rn_subreddits[doc[3]] += 1
                else:
                    rn_subreddits[doc[3]] = 1
                seen.append(doc[3])
        else:
            raise Exception("Invalid data!")

    for sr in all_sr:
        print(sr)
        if sr in spreader_subreddits.keys():
            print("Spreader: " + str(spreader_subreddits[sr]))
        else:
            print("Spreader: 0")
        if sr in rn_subreddits.keys():
            print("Real News: " + str(rn_subreddits[sr]))
        else:
            print("Real News: 0")

    print(all_sr)

def data_stats():
    source_frame = RedditUserDataset.load_from_file('data/core_dataset/reddit_corpus_final_balanced.gzip', compression='gzip')
    users = pd.read_pickle('data/core_dataset/reddit_corpus_final_balanced.gzip', compression='gzip')['user_id']
    source_frame.timeframed_documents((datetime.date(2020, 1, 1), datetime.date(2021, 4, 30)), inplace=True)
    spreader_docs = 0
    spreader_num = 0
    checker_docs = 0
    checker_num = 0
    doc_sum = 0
    real_docs = 0
    fake_docs = 0
    for index, row in source_frame.data_frame.iterrows():
        if index not in users or len(row['documents']) == 0:
            continue
        if row['fake_news_spreader'] == 1:
            spreader_num += 1
            spreader_docs += len(row['documents'])
        else:
            checker_num += 1
            checker_docs += len(row['documents'])

        doc_sum += len(row['documents'])

    print('Spreader Num: ' + str(spreader_num))
    print('Spreader Docs: ' + str(spreader_docs/spreader_num))

    print('Checker Num: ' + str(checker_num))
    print('Checker Docs: ' + str(checker_docs / checker_num))

    print(doc_sum)
    print(real_docs)
    print(fake_docs)

if __name__ == '__main__':
    #baselines()
    data_stats()
    #res = df.load_social_graph((str(datetime.date(2020, 1, 1)), str(datetime.date(2020, 1, 31))), inplace=False)
    #res.store_to_file('data/social_test_timeframe.gzip')
