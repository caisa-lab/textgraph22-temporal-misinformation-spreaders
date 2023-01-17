from data_collection.reddit_user_dataset import RedditUserDataset
from classification.feature_computing import SavedPostBertEmbedder
import datetime
import numpy as np


def get_embeddings(frame, embed_mode='avg'):
    group_by = frame.groupby(by='embedding_file')
    embedding_dict = {}
    for group_tuple in group_by:
        embedder = SavedPostBertEmbedder(group_tuple[0])
        for index, row in group_tuple[1].iterrows():
            if len(row['documents']) == 0:
                embedding_dict[index] = np.zeros(1024)
                continue
            try:
                timestamp_dict = {}
                for doc in row['documents']:
                    post_date = doc[2]
                    if isinstance(post_date, str):
                        post_date = datetime.strptime(doc[2], '%Y-%m-%d %H:%M:%S')
                    timestamp_dict[doc[0]] = post_date
                embedding_dict[index] = embedder.embed_user([doc[0] for doc in row['documents']], mode=embed_mode,
                                                            timestamp_map=timestamp_dict)
            except Exception as e:
                print("Exception while embedding user " + str(index))
                print(e)
    embedder = None
    return embedding_dict

timeframes = [(datetime.date(2020, 2, 1), datetime.date(2020, 2, 29)),
              (datetime.date(2021, 1, 28), datetime.date(2021, 2, 28)),
              (datetime.date(2021, 3, 1), datetime.date(2021, 3, 31))]


source_dataset = RedditUserDataset.load_from_file("data/core_dataset/core_dataset.gzip", compression='gzip')

frames = [source_dataset.timeframed_documents(span, inplace=False).data_frame for span in timeframes]

user_embedding_dict = {}

for index, row in source_dataset.data_frame.iterrows():
    user_embedding_dict[index] = []

for frame in frames:
    embs = get_embeddings(frame)
    for index, row in source_dataset.data_frame.iterrows():
        user_embedding_dict[index].append(embs[index])