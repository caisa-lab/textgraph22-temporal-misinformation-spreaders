from classification.feature_computing import SavedPostBertEmbedder
from data_collection.reddit_user_dataset import convert_timeframes_to_model_input, WindowGraphData, RedditUserDataset
import pickle as pkl
import datetime
import random
import gzip
import time
import os
from os import listdir
from utils.file_sort import path_sort
from argparse import ArgumentParser
import argparse
import numpy as np
from os.path import isfile, join
from random import randrange
import json
from utils.utils import *
from utils.file_sort import path_sort
import torch
from tqdm import tqdm
from data_collection.reddit_user_dataset import WindowGraphData


parser = ArgumentParser()
parser.add_argument("--base_dataset", dest="base_dataset_path", required=True, type=str)
parser.add_argument("--source_frames", dest="source_frames_path", required=True, type=str)
parser.add_argument("--filename", dest="filename", required=True, type=str)

args = parser.parse_args()

# Build ground truth
base_dataset = RedditUserDataset.load_from_file(args.base_dataset_path, compression='gzip')

ground_truth = {}
for index, row in base_dataset.data_frame.iterrows():
    ground_truth[index] = row['fake_news_spreader']

source_graph_descriptor = pkl.load(
    gzip.open(os.path.join(args.source_frames_path, 'source_graph_descriptor.data'), 'rb'))

print(source_graph_descriptor)
doc_embedding_file_path = source_graph_descriptor['embedding_file_path']
if 'embedding_file_header' in source_graph_descriptor:
    doc_embedding_file_header = source_graph_descriptor['embedding_file_header']
else:
    doc_embedding_file_header = 'embedding_file'
embed_mode = source_graph_descriptor['embed_mode']

threshold = 0.8

timeframed_dataset = []
doc_amount_avgs = []
source_frame_dir = args.source_frames_path


for graph in tqdm(path_sort(
        [join(source_frame_dir, f) for f in listdir(source_frame_dir) if isfile(join(source_frame_dir, f))])):
    if "source_graph_descriptor.data" in graph:
        continue
    print(graph)
    timeframe_ds = RedditUserDataset.load_from_instance_file(graph)
    timeframe_ds.shorten_similarity_triplet_list(threshold)
    timeframed_dataset.append(timeframe_ds)
    doc_sum = 0
    users = 0
    for index, row in RedditUserDataset.load_from_instance_file(graph).data_frame.iterrows():
        users += 1
        doc_sum += row['num_docs']
    doc_amount_avgs.append(doc_sum / users)


precomputed_features = {}

for index, row in base_dataset.data_frame.iterrows():
    precomputed_features[index] = []
    
for frame in timeframed_dataset:
    print('Precomputing features for timeframe...')
    feature_map = frame.compute_features(doc_embedding_file_path, doc_embedding_file_header, embed_mode=embed_mode)
    for index, feature in feature_map.items():
        if index in precomputed_features.keys():
            precomputed_features[index].append(feature_map[index])


name = args.filename
with open('../data/stored_embeddings/'+ name + '.pkl', 'wb') as f:
    pkl.dump(precomputed_features, f, pkl.HIGHEST_PROTOCOL)