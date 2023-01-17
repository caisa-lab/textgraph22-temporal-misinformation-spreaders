import pickle as pkl
from utils.train_utils import get_embeddings_dict_from_path
import gzip

source_files = ['data/stored_embeddings/doc_embeddings_1.txt'] 
        #'data/stored_embeddings/doc_embeddings_2.txt',
        #'data/stored_embeddings/doc_embeddings_3.txt',
        #'data/stored_embeddings/doc_embeddings_4.txt',
        #'data/stored_embeddings/doc_embeddings_5.txt']

for emb_file in source_files:
    print(emb_file)
    emb_dict = get_embeddings_dict_from_path(emb_file)
    target = emb_file.replace('txt', 'gzip')
    print(target)
    with gzip.open(target, 'w') as output:
        pkl.dump(emb_dict, output)
    emb_dict = None    
