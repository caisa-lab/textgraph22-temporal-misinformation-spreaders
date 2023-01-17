from data_collection.reddit_user_dataset import RedditUserDataset
from classification.feature_computing import SavedPostBertEmbedder
from utils.file_sort import path_sort
import os
import datetime
import time
import pickle as pkl
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser
import argparse
import gzip
import sys
from tqdm import tqdm
from pathlib import Path
from utils.train_utils import *



base_dataset_path = 'data/core_dataset/reddit_corpus_final_balanced.gzip'
embedding_file_path = 'data/stored_embeddings/'
embedding_header = 'embedding_file'
delta_days = 30
offset_days = 30
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2021, 4, 30)
embed_mode = 'avg'

curr_date = start_date
timeframes = []
while curr_date + datetime.timedelta(days=delta_days) < end_date:
    print(curr_date)
    timeframes.append((curr_date, curr_date + datetime.timedelta(days=delta_days)))
    curr_date = curr_date + datetime.timedelta(days=offset_days)

base_dataset = RedditUserDataset.load_from_file(base_dataset_path, compression='gzip')

# tf = timeframes[0]
for index, tf in tqdm(enumerate(timeframes)):
    framed = base_dataset.timeframed_documents(tf, inplace=False)
    start = time.time()

    embeddings = framed.compute_features(embedder, embed_mode=embed_mode, num_features=768)

    res_path = os.path.join(embedding_file_path,
                            'user_embeddings/' + embed_mode + '/delta_'+str(delta_days)+'/')

    Path(res_path).mkdir(parents=True, exist_ok=True)

    emb_output_file = res_path + 'user_em_' + embed_mode + '_delta_'+str(delta_days)+'_' + str(index) + '.gzip'

    write_embeddings(embeddings, emb_output_file)


    end = time.time()
    print("Elapsed time:" + str(end - start))



