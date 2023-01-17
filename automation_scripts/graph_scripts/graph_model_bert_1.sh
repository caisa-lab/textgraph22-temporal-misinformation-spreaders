python ../src/training_graph.py --patience=40 \
--run_id='bertGAT_ling_ts' \
--sample_dir='../data/twitter_dataset/model_samples_avg/ts_bert_delta30/'  \
--result_dir='../results/graph_model/twitter' \
--checkpoint_dir='../results/to_check/' \
--max_epochs=50 \
--learning_rate=5e-5 \
--nheads=4 \
--dropout=0.2 \
--nhid_graph=256 \
--nhid_gru=128 \
--users_dim=768 \
--graph_layer='true' \
--rnn_layer='gru' \
--att_layer='true' \
--gnn='gat' \
--merge_samples='false' |& tee ../logs/graph_model_main.txt
 
