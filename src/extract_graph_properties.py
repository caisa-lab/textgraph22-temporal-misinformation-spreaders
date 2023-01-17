from data_collection.reddit_user_dataset import RedditUserDataset
from classification.feature_computing import Embedder
import pickle as pkl
import os
from os import listdir
from utils.file_sort import path_sort
from os.path import isfile, join
import json
from tqdm import tqdm
import gzip


class GraphNumbers:
    def __init__(self, edges, nodes, fns, rns, m2m, m2r, r2r):
        self.edges = edges
        self.nodes = nodes
        self.fns = fns
        self.rns = rns
        self.m2m = m2m
        self.m2r = m2r 
        self.r2r = r2r
        
    def print_attr(self):
        print(self.__dict__)
        
        
if __name__ == '__main__':
    base_dataset_path='../data/reddit_dataset/reddit_corpus_unbalanced_filtered.gzip' 
    base_dataset = RedditUserDataset.load_from_file(base_dataset_path, compression='gzip')
    
    source_frame_dir = '../data/reddit_dataset/linguistic/cosine/avg/usr2vec_delta30_new/source'
    source_graph_descriptor = pkl.load(
        gzip.open(os.path.join(source_frame_dir, 'source_graph_descriptor.data'), 'rb'))
    print(source_graph_descriptor)
    doc_embedding_file_path= source_graph_descriptor['embedding_file_path']
    embed_type= source_graph_descriptor['embed_type']
    dim=source_graph_descriptor['dim']
    embedder = Embedder([doc_embedding_file_path], embed_type, dim)
    threshold = 0.8
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
    
    
    
    threshold = 0.8
    numbers_per_month = []

    for tf_d in tqdm(timeframed_dataset):
        df = tf_d.data_frame
        n_edges = 0
        nodes = set()
        n_fns = set()
        n_rns = set()
        n_m2m = 0
        n_m2r = 0
        n_r2r = 0
        
        for similarities in tf_d.similarity_triplets:
            user1, user2, sim = similarities
            if sim > threshold:
                n_edges += 1
                nodes.add(user1)
                nodes.add(user2)
                
                label1 = df[df['user_id'] == user1]['fake_news_spreader'].values[0]
                label2 = df[df['user_id'] == user2]['fake_news_spreader'].values[0]
                
                if label1 == 1:
                    n_fns.add(user1)
                else:
                    n_rns.add(user1)
                    
                if label2 == 1:
                    n_fns.add(user2)
                else:
                    n_rns.add(user2)
                
                if label1 != label2:
                    n_m2r += 1
                elif label1 == 1 and label2 == 1:
                    n_m2m += 1
                elif label1 == 0 and label2 == 0:
                    n_r2r += 1
                else:
                    print(label1, label2)
                    raise Exception("Wrong")
                
        numbers_per_month.append(GraphNumbers(n_edges, len(nodes), len(n_fns), len(n_rns), n_m2m, n_m2r, n_r2r))
        

    with open('../data/usr2vec_semantic.json', 'w') as f:
        json.dump([g.__dict__ for g in numbers_per_month], f)