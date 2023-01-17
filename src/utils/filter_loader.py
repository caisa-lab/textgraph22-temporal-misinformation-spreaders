import pandas as pd
import sys
import random
import re
from data_collection.reddit_user_dataset import RedditUserDataset

def contains_link(doc):
    return len(re.findall(
        r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
        doc)) > 0


all_users_file = 'data/core_dataset/core_dataset.gzip'
ds = RedditUserDataset.load_from_file(all_users_file, compression='gzip')
data = RedditUserDataset.load_from_file(all_users_file, compression='gzip').data_frame
data['rn_amounts'] = data['amounts'].apply(lambda x: x[0])
data['fn_amounts'] = data['amounts'].apply(lambda x: x[1])
data['fn_rn_ratio'] = data['fn_amounts']/data['rn_amounts']
filter_mode = int(sys.argv[1])
if filter_mode == 1:
    print('Running the first filter')
    n_verified_fn_spreaders = len(data[(data['fn_rn_ratio']>=1)]['fn_amounts'])
    df_verified_fn_spreaders = data[(data['fn_rn_ratio']>=1)]
    df_verified_rn_spreaders = data[data['fake_news_spreader']==0].sort_values(by=['rn_amounts'], ascending = False).head(n_verified_fn_spreaders)
    data = pd.concat([df_verified_rn_spreaders, df_verified_fn_spreaders], axis=0)
    print(data)
elif filter_mode == 2:
    print('Running the second filter')
    n_verified_fn_spreaders = int(len(data[data['fake_news_spreader']==1])/2)
    df_verified_fn_spreaders = data[data['fake_news_spreader']==1].sort_values(by=['fn_amounts', 'fn_rn_ratio'], ascending = False).head(n_verified_fn_spreaders)
    df_verified_rn_spreaders = data[data['fake_news_spreader']==0].sort_values(by=['rn_amounts'], ascending = False).head(n_verified_fn_spreaders)
    save_path = 'data/persistence_results/verified_monthly_graphs_v2/'
    data = pd.concat([df_verified_rn_spreaders, df_verified_fn_spreaders], axis=0)
    print(data)
elif filter_mode == 3:
    print('Running the third filter')
    keep_ids = []
    for index, row in data.iterrows():
        if len(row['documents']) >= 150:
            keep_ids.append(index)
    filtered = ds.filter_user_ids(keep_ids, inplace=False).data_frame

    new_docs = []
    for index, row in filtered.iterrows():
        docs = row['documents']
        sampled = random.sample(docs, 150)
        new_docs.append(sampled)

    filtered['documents'] = new_docs
    df_verified_fn_spreaders = filtered[filtered['fake_news_spreader'] == 1]
    df_verified_rn_spreaders = filtered[filtered['fake_news_spreader'] == 0]

    balance_amount = min(len(df_verified_fn_spreaders), len(df_verified_rn_spreaders))

    filtered_ds = RedditUserDataset(filtered)
    filtered_ds.fit_label_amount('fake_news_spreader', {0: balance_amount, 1: balance_amount})
    data = filtered_ds.data_frame
elif filter_mode == 4:
    print('Running the fourth filter')
    filtered_docs = []
    for index, row in data.iterrows():
        filtered = [doc for doc in row['documents'] if doc[3] == "r/politics"]
        filtered_docs.append(filtered)
    data['documents'] = filtered_docs
    print(data)
elif filter_mode == 5:
    print('Running the fourth filter')
    filtered_docs = []
    for index, row in data.iterrows():
        filtered = [doc for doc in row['documents'] if not contains_link(doc[1])]
        filtered_docs.append(filtered)
    data['documents'] = filtered_docs
    print(data)
elif filter_mode == 6:
    print('Running the sixth filter')
    filtered_docs = []
    for index, row in data.iterrows():
        filtered = [doc for doc in row['documents'] if not contains_link(doc[1])]
        filtered = [doc for doc in filtered if doc[3] == "r/politics"]
        filtered_docs.append(filtered)
    data['documents'] = filtered_docs

    keep_ids = []
    for index, row in data.iterrows():
        if len(row['documents']) >= 80:
            keep_ids.append(index)
    ds = RedditUserDataset(data)
    filtered = ds.filter_user_ids(keep_ids, inplace=False).data_frame

    new_docs = []
    for index, row in filtered.iterrows():
        docs = row['documents']
        sampled = random.sample(docs, 80)
        new_docs.append(sampled)

    filtered['documents'] = new_docs
    df_verified_fn_spreaders = filtered[filtered['fake_news_spreader'] == 1]
    df_verified_rn_spreaders = filtered[filtered['fake_news_spreader'] == 0]

    balance_amount = min(len(df_verified_fn_spreaders), len(df_verified_rn_spreaders))

    filtered_ds = RedditUserDataset(filtered)
    filtered_ds.fit_label_amount('fake_news_spreader', {0: balance_amount, 1: balance_amount})
    data = filtered_ds.data_frame
    for index, row in data.iterrows():
        print(len(row['documents']))
        print(row['documents'])
    print(len(data))
data.to_pickle('data/filtered_datasets/filter_' + str(filter_mode) + '.gzip', compression='gzip')