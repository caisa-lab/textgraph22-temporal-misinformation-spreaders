from classification.feature_computing import SavedPostBertEmbedder
from data_collection.reddit_user_dataset import convert_timeframes_to_model_input, WindowGraphData, RedditUserDataset
import networkx as nx
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
import torch
from tqdm import tqdm
from data_collection.reddit_user_dataset import WindowGraphData


def add_connections(graph_data, source, target, temp_tuples):
    for i in range(len(graph_data[0])):
        connection = (graph_data[0][i], graph_data[1][i])

        if connection not in temp_tuples:
            source.append(connection[0])
            target.append(connection[1])
            temp_tuples.add(connection)
    
    return source, target, temp_tuples
            
def merge_samples(sample_1, sample_2):
    assert len(sample_1.graph_data) == len(sample_2.graph_data), print("Two samples do not have same size, cannot merge")
    assert sample_1.n_feat == sample_2.n_feat
    
    graph_data = []
    
    for i in range(len(sample_1.graph_data)):
        source, target, temp_tuples = add_connections(sample_1.graph_data[i], [], [], set())
        source, target, _ = add_connections(sample_2.graph_data[i], source, target, temp_tuples)

        
        
        edges = torch.cat([torch.tensor(source).unsqueeze(-1), torch.tensor(target).unsqueeze(-1)], dim=1).permute(1, 0)
        graph_data.append(edges)
        
    
    return WindowGraphData(sample_1.n_classes, sample_1.window, sample_1.n_feat, sample_1.n_nodes,
                           graph_data, sample_1.features, sample_1.labels, sample_1.user_index)
        

def norm(input_list):
    norm_list = list()

    if isinstance(input_list, list):
        sum_list = sum(input_list)

        for value in input_list:
            tmp = value / sum_list
            norm_list.append(tmp)

    return norm_list


def chunk_data(to_chunk, amount):
    chunk_size = len(to_chunk) // amount
    chunk_size = max(1, chunk_size)
    return list((to_chunk[i:i + chunk_size] for i in range(0, len(to_chunk), chunk_size)))[:amount]


def fold_data(chunked_data: [[]], fold_index: int):
    train_data = []
    val_data = []
    test_data = []

    for index in range(0, len(chunked_data)):
        if index == fold_index:
            test_data.extend(chunked_data[index])
        elif index == (fold_index + 1) % len(chunked_data):
            val_data.extend(chunked_data[index])
        else:
            train_data.extend(chunked_data[index])

    return train_data, val_data, test_data


parser = ArgumentParser()
parser.add_argument("--base_dataset", dest="base_dataset_path", required=True, type=str)
parser.add_argument("--source_frames", dest="source_frames_path", required=True, type=str)
parser.add_argument("--social_sf", dest="social_sf_path", required=True, type=str)
parser.add_argument("--sample_dir", dest="sample_dir", required=True, type=str)
parser.add_argument("--n_users", dest="n_users", default=300, type=int)
parser.add_argument("--n_train_samples", dest="n_train_samples", default=800, type=int)
parser.add_argument("--n_val_samples", dest="n_val_samples", default=200, type=int)
parser.add_argument("--n_test_samples", dest="n_test_samples", default=1, type=int)
parser.add_argument("--threshold", dest="threshold", default=0.8, type=float)
parser.add_argument("--percentage", dest="percentage", default=1, type=float)
parser.add_argument("--root_seed", dest="root_seed", default=1337, type=int)
parser.add_argument("--train_min_index", dest="train_min_index", required=True, type=int)
parser.add_argument("--train_max_index", dest="train_max_index", required=True, type=int)
parser.add_argument("--val_min_index", dest="val_min_index", required=True, type=int)
parser.add_argument("--val_max_index", dest="val_max_index", required=True, type=int)
parser.add_argument("--test_min_index", dest="test_min_index", required=True, type=int)
parser.add_argument("--test_max_index", dest="test_max_index", required=True, type=int)
parser.add_argument("--do_user_split", dest="do_user_split", default=False, type=str2bool)
parser.add_argument("--user_fold_amount", dest="user_fold_amount", type=int, default=4)
parser.add_argument("--user_fold_index", dest="user_fold_index", type=int, default=0)
parser.add_argument("--random_features", dest="random_features", required=False, default=False, type=str2bool)

args = parser.parse_args()
base_dataset = RedditUserDataset.load_from_file(args.base_dataset_path, compression='gzip')

if args.do_user_split and (args.user_fold_index > args.user_fold_amount - 1 or args.user_fold_index < 0):
    raise Exception("Invalid user_fold_index!")

# Build ground truth
ground_truth = {}
for index, row in base_dataset.data_frame.iterrows():
    ground_truth[index] = row['fake_news_spreader']

source_frame_dir = args.source_frames_path
social_sf_dir = args.social_sf_path
target_dir = args.sample_dir

print("Running model dataloader with target dir {}".format(target_dir))

if not os.path.exists(os.path.join(target_dir, 'train_samples/')):
    os.makedirs(os.path.join(target_dir, 'train_samples/'))
if not os.path.exists(os.path.join(target_dir, 'val_samples/')):
    os.makedirs(os.path.join(target_dir, 'val_samples/'))
if not os.path.exists(os.path.join(target_dir, 'test_samples/')):
    os.makedirs(os.path.join(target_dir, 'test_samples/'))

train_min_index = args.train_min_index
train_max_index = args.train_max_index
val_min_index = args.val_min_index
val_max_index = args.val_max_index
test_min_index = args.test_min_index
test_max_index = args.test_max_index
n_users = args.n_users
n_train_samples = args.n_train_samples
n_val_samples = args.n_val_samples
n_test_samples = args.n_test_samples
threshold = args.threshold
percentage = args.percentage
ROOT_SEED = args.root_seed


source_graph_descriptor = pkl.load(
    gzip.open(os.path.join(args.source_frames_path, 'source_graph_descriptor.data'), 'rb'))

if 'embedding_file_header' in source_graph_descriptor:
    doc_embedding_file_header = source_graph_descriptor['embedding_file_header']
else:
    doc_embedding_file_header = 'embedding_file'
doc_embedding_file_path = source_graph_descriptor['embedding_file_path']
embed_mode = source_graph_descriptor['embed_mode']

dataset_descriptor = {}
dataset_descriptor['n_users'] = n_users
dataset_descriptor['n_train_samples'] = n_train_samples
dataset_descriptor['n_val_samples'] = n_val_samples
dataset_descriptor['n_test_samples'] = n_test_samples
dataset_descriptor['threshold'] = threshold
dataset_descriptor['percentage'] = percentage
dataset_descriptor['root_seed'] = ROOT_SEED
dataset_descriptor['base_dataset'] = args.base_dataset_path
dataset_descriptor['source_frames'] = args.source_frames_path
dataset_descriptor['social_sf'] = args.social_sf_path
dataset_descriptor['embed_mode'] = embed_mode
dataset_descriptor['do_user_split'] = args.do_user_split
dataset_descriptor['user_fold_amount'] = args.user_fold_amount
dataset_descriptor['user_fold_index'] = args.user_fold_index
dataset_descriptor['time_splits'] = [[train_min_index, train_max_index],
                                     [val_min_index, val_max_index],
                                     [test_min_index, test_max_index]]
dataset_descriptor['doc_embedding_file_path'] = doc_embedding_file_path
dataset_descriptor['doc_embedding_file_header'] = doc_embedding_file_header
dataset_descriptor['graph_type'] = 'merged'

json.dump(dataset_descriptor, open(os.path.join(target_dir, 'dataset_descriptor.json'), 'w'))


random.seed(ROOT_SEED)
np.random.seed(ROOT_SEED)
sampling_seeds = [int(random.uniform(0, 1000000)) for i in range(n_train_samples + n_test_samples + n_val_samples)]

timeframed_dataset = []
doc_amount_avgs = []
for graph in path_sort(
        [join(source_frame_dir, f) for f in listdir(source_frame_dir) if isfile(join(source_frame_dir, f))]):
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

social_dataset = []
social_doc_amount_avgs = []
from utils.file_sort import path_sort

for graph in tqdm(path_sort(
        [join(social_sf_dir, f) for f in listdir(social_sf_dir) if isfile(join(social_sf_dir, f))])):
    if "source_graph_descriptor.data" in graph:
        continue
    print(graph)
    social_ds = RedditUserDataset.load_from_instance_file(graph)
    social_ds.shorten_similarity_triplet_list(threshold)
    social_dataset.append(social_ds)
    doc_sum = 0
    users = 0
    for index, row in RedditUserDataset.load_from_instance_file(graph).data_frame.iterrows():
        users += 1
        doc_sum += row['num_docs']
    social_doc_amount_avgs.append(doc_sum / users)

# Split user
if args.do_user_split:
    user_ids = []
    for index, row in base_dataset.data_frame.iterrows():
        user_ids.append(row['user_id'])

    chunked = chunk_data(user_ids, args.user_fold_amount)
    train_ids, val_ids, test_ids = fold_data(chunked, args.user_fold_index)
    print(len(train_ids))
    print(len(val_ids))
    print(len(test_ids))

    # Split validation
    for uid in train_ids:
        if uid in val_ids:
            raise Exception("Invalid split!")
        if uid in test_ids:
            raise Exception("Invalid split")

    for uid in val_ids:
        if uid in test_ids:
            raise Exception("Invalid split!")

    train_sample_frame = base_dataset.filter_user_ids(train_ids, inplace=False).data_frame
    print(len(train_sample_frame))
    val_sample_frame = base_dataset.filter_user_ids(val_ids, inplace=False).data_frame
    print(len(val_sample_frame))
    test_sample_frame = base_dataset.filter_user_ids(test_ids, inplace=False).data_frame
    print(len(test_sample_frame))
else:
    print("Users are not being split...")
    train_sample_frame = base_dataset.data_frame
    val_sample_frame = base_dataset.data_frame
    test_sample_frame = base_dataset.data_frame

precomputed_features = {}

for index, row in base_dataset.data_frame.iterrows():
    precomputed_features[index] = []

if not args.random_features:
    # TODO: compute features from document embeddings files to timeframed_dataset.
    for frame in timeframed_dataset:
        print('Precomputing features for timeframe...')
        feature_map = frame.compute_features(doc_embedding_file_path, doc_embedding_file_header, embed_mode=embed_mode)
        for index, feature in feature_map.items():
            if index in precomputed_features.keys():
                precomputed_features[index].append(feature_map[index])
else:
    for frame in timeframed_dataset:
        print('Precomputing random features...')
        for index, row in frame.data_frame.iterrows():
            precomputed_features[index].append(torch.tensor(np.random.uniform(low=-1.5, high=1.5, size=768)))

seed_counter = -1

start = time.time()

for n in range(n_train_samples):
    seed_counter += 1
    sample_ids = train_sample_frame.sample(n=n_users, random_state=sampling_seeds[seed_counter])['user_id']
    sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in timeframed_dataset]
    social_sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in social_dataset]

    start_frame = train_min_index
    train_window = train_max_index - train_min_index
    sample_frames = sample_frames[start_frame:start_frame + train_window]
    social_sample_frames = social_sample_frames[start_frame:start_frame + train_window]
    
    sample_frames = [
        tf.build_graph_column_precomputed(threshold=threshold, percentage=percentage, inplace=False).data_frame for
        index, tf in
        enumerate(sample_frames)]
  
    social_sample_frames = [
        tf.data_frame for tf in social_sample_frames
    ]
    
    sample = convert_timeframes_to_model_input(sample_frames, {k: v[start_frame:start_frame + train_window] for k, v in
                                                               precomputed_features.items()}, ground_truth)
    
    social_sample = convert_timeframes_to_model_input(social_sample_frames, {k: v[start_frame:start_frame + train_window] for k, v in
                                                               precomputed_features.items()}, ground_truth)
   
    sample = merge_samples(sample, social_sample)
    
    pkl.dump(sample, gzip.open(os.path.join(target_dir, 'train_samples/') + 'sample_' + str(n) + '.data', 'wb'))

end = time.time()
print("Elapsed time:" + str(end - start))

start = time.time()
for n in range(n_val_samples):
    seed_counter += 1
    sample_ids = val_sample_frame.sample(n=n_users, random_state=sampling_seeds[seed_counter])['user_id']
    try:
        sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in timeframed_dataset]  
    except IndexError as e:
        print(e)
        print(n)
        print(len(timeframed_dataset))
        print(sample_ids)
        
    social_sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in social_dataset]

    start_frame = val_min_index
    val_window = val_max_index - val_min_index

    sample_frames = sample_frames[start_frame:start_frame + val_window]
    social_sample_frames = social_sample_frames[start_frame:start_frame + val_window]

    sample_frames = [
        tf.build_graph_column_precomputed(threshold=threshold, percentage=percentage, inplace=False).data_frame for
        index, tf in
        enumerate(sample_frames)]

    social_sample_frames = [tf.data_frame for tf in social_sample_frames]

    sample = convert_timeframes_to_model_input(sample_frames, {k: v[start_frame:start_frame + val_window] for k, v in
                                                               precomputed_features.items()}, ground_truth)
    social_sample = convert_timeframes_to_model_input(social_sample_frames, {k: v[start_frame:start_frame + train_window] for k, v in
                                                        precomputed_features.items()}, ground_truth)

    sample = merge_samples(sample, social_sample)
    sample.print_shapes()
    pkl.dump(sample, gzip.open(os.path.join(target_dir, 'val_samples/') + 'sample_' + str(n) + '.data', 'wb'))

end = time.time()
print("Elapsed time:" + str(end - start))

tart = time.time()
for n in range(n_test_samples):
    seed_counter += 1
    sample_ids = test_sample_frame['user_id']
    sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in timeframed_dataset]
    social_sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in social_dataset]

    start_frame = test_min_index
    test_window = test_max_index - test_min_index

    sample_frames = sample_frames[start_frame:]
    social_sample_frames = social_sample_frames[start_frame:]

    sample_frames = [
        tf.build_graph_column_precomputed(threshold=threshold, percentage=1, inplace=False).data_frame for
        index, tf in
        enumerate(sample_frames)]
    social_sample_frames = [tf.data_frame for tf in social_sample_frames]
    
    sample = convert_timeframes_to_model_input(sample_frames, {k: v[start_frame:] for k, v in
                                                               precomputed_features.items()}, ground_truth)
    social_sample = convert_timeframes_to_model_input(social_sample_frames, {k: v[start_frame:start_frame + train_window] for k, v in
                                                        precomputed_features.items()}, ground_truth)
    sample = merge_samples(sample, social_sample)

    sample.print_shapes()
    pkl.dump(sample, gzip.open(os.path.join(target_dir, 'test_samples/') + 'sample_' + str(n) + '.data', 'wb'))

end = time.time()
print("Elapsed time:" + str(end - start))


dataset_descriptor['source_frame_descriptor'] = source_graph_descriptor

for key, val in dataset_descriptor.items():
    if isinstance(val, dict):
        dataset_descriptor[key] = {k: convert_value(v) for k,v in val.items()}
    else:
        dataset_descriptor[key] = val

json.dump(dataset_descriptor, open(os.path.join(target_dir, 'dataset_descriptor.json'), 'w'))




