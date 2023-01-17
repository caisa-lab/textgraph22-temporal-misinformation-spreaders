import os
import pickle as pkl
from data_collection.reddit_user_dataset import RedditUserDataset

target_dir = 'data/core_dataset/linguistic/cosine/avg/delta30off15/source/'

for graph in os.listdir(target_dir):
    if '.pkl' in graph:
        graph_file = os.path.join(target_dir, graph)
        print(graph_file)
        ds = RedditUserDataset.load_from_instance_file(graph_file)
        new_col = []
        for index, row in ds.data_frame.iterrows():
            emb = row['embedding_file']
            mod_emb = emb.replace('.txt', '.gzip')
            new_col.append(mod_emb)
        ds.data_frame['gzip_embeddings'] = new_col
        ds.store_instance_to_file(graph_file)
