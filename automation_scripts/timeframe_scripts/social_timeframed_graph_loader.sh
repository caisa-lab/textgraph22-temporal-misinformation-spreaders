python ../src/timeframed_graph_loader.py \
--gen_source_graphs=True \
--delta_days=30 \
--offset_days=30 \
--graph_type='social' \
--path='../data/reddit_dataset/social/bert_delta30/' \
--base_dataset='../data/reddit_dataset/reddit_corpus_unbalanced_filtered.gzip' \
--doc_embedding_file_path='../data/embeddings/bert/' \
--embed_type='bert' \
--merge_liwc='false' \
--dim=768 \
--user_interaction_file_dir='../data/social_graph_data/' \
--embed_mode='avg' |& tee ../logs/social_timeframed_graph_loader.txt