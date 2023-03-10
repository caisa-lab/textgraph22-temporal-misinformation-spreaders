python ../src/model_dataloader.py \
--n_users=200 \
--n_train_samples=1000 \
--n_val_samples=200 \
--base_dataset='../data/reddit_dataset/reddit_corpus_unbalanced_filtered.gzip' \
--source_frames='../data/reddit_dataset/social/bert_delta30/source' \
--sample_dir='../data/reddit_dataset/model_samples_social/mix_bert_delta30_new/' \
--user_ids='../data/reddit_dataset/user_splits/binary_unbalanced' \
--threshold=0.8 \
--do_user_split='t' \
--percentage=1 \
--train_min_index=0 \
--train_max_index=8 \
--val_min_index=8 \
--val_max_index=16 \
--test_min_index=8 \
--test_max_index=16 |& tee ../logs/social_dataloader_timesplit.txt