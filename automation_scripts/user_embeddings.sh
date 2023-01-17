#!/bin/bash
python ../src/user_embeddings.py \
--bert_model='all-mpnet-base-v2' \
--in_file='../data/twitter_dataset_filtered_cleaned.gzip' \
--embeddings_folder='../data/stored_embeddings'