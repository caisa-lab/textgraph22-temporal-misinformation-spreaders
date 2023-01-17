python ../src/timeframed_graph_loader.py \
--gen_source_graphs=True \
--delta_days=480 \
--offset_days=480 \
--path='../data/reddit_dataset/linguistic/cosine/avg/liwc_delta480/' \
--base_dataset='../data/reddit_dataset/reddit_corpus_balanced_filtered.gzip' \
--doc_embedding_file_path='../data/embeddings/psycho/' \
--embed_type='liwc' \
--embed_mode='avg' |& tee ../logs/timeframed_graph_loader.txt
