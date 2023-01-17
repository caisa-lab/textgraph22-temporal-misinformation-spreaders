python ../src/training_graph.py --patience=40 \
--run_id='vecGCN_social_mix_split' \
--sample_dir='../data/reddit_dataset/model_samples_social/mix_usr2vec_delta30_new' \
--result_dir='../results/graph_model/15_feb' \
--checkpoint_dir='../results/to_check/' \
--max_epochs=100 \
--learning_rate=5e-5 \
--nheads=4 \
--dropout=0.2 \
--nhid_graph=100 \
--nhid_gru=50 \
--users_dim=200 \
--graph_layer='true' \
--rnn_layer='gru' \
--att_layer='true' \
--gnn='gcn' \
--merge_samples='false' |& tee ../logs/graph_model_usr2vec_social.txt
 