from utils.train_utils import *
from nltk import tokenize
import pandas as pd
import pickle as pkl
import re
from sentence_transformers import SentenceTransformer
from constants import *
import argparse
import os
import sys 
import logging


parser = argparse.ArgumentParser(description='User embeddings!')
parser.add_argument('--in_file', help='in file')
parser.add_argument('--embeddings_folder', help='user out file')
parser.add_argument('--bert_model', help='Bert model')

args = parser.parse_args()

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    
    source_frame = pd.read_pickle(args.in_file, compression='gzip')
    root_dir = os.path.join(args.embeddings_folder, args.bert_model)
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    model = SentenceTransformer(args.bert_model).to(DEVICE)

    docs_embeddings = {}
    user_embeddings = {}

    logger.info(f'Using {DEVICE}')
    logger.info("Computing embeddings...")

    ctr = 0
    for index, row in source_frame.iterrows():
        ctr += 1
        docs = row['documents']
        user_id = row['user_id']
        user_emb = None
        
        for doc_id, doc, _, _ in docs:
            temp_embeddings = model.encode(process_tweet(doc), show_progress_bar=False)
            docs_embeddings[doc_id] = temp_embeddings
            
            if user_emb is None:
                user_emb = temp_embeddings
            else:
                user_emb += temp_embeddings
        
        user_embeddings[user_id] = (1 / len(docs)) * user_emb
        
        if ctr % 100 ==0:
            logger.info(f'User number {ctr}, with index {index}')

        if ctr % 1000 == 0:
           idx = int(ctr / 1000)
           user_file = f'user_embeddings_{idx}.txt'
           doc_file = f'doc_embeddings_{idx}.txt'

           write_embeddings(user_embeddings, os.path.join(root_dir, user_file))
           write_embeddings(docs_embeddings, os.path.join(root_dir, doc_file))
           user_embeddings.clear()
           docs_embeddings.clear()
    


    user_file = f'user_embeddings_{idx+1}.txt'
    doc_file = f'doc_embeddings_{idx+1}.txt'

    write_embeddings(user_embeddings, os.path.join(root_dir, user_file))
    write_embeddings(docs_embeddings, os.path.join(root_dir, doc_file))
