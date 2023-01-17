import pandas as pd
#from bert_serving.client import BertClient
from nltk import tokenize
import pickle as pkl
import re
from sentence_transformers import SentenceTransformer

source_frame = pd.read_pickle('data/twitter_dataset/twitter_dataset.gzip', compression='gzip')
res_path = 'data/stored_embeddings/twitter_dataset_embeddings'

# Remember to start bert-as-service server for using this embedding
#bert_client = BertClient()

model = SentenceTransformer('stsb-roberta-large')
model.max_seq_length = 512

def doc_to_bert_interpretable(doc):
    """
    Helper method for transferring a given document
    into a bert-interpretable string, where senteces are
    separated by '|||'
    :param doc: The document to transform
    :return: The bert-interpretable string
    """
    sentences = tokenize.sent_tokenize(doc)
    res = ""
    for index, sentence in enumerate(sentences):
        res += sentence
        if index < len(sentences) - 1:
            res += ' ||| '
    return res

def mask_links(post):
    links = re.findall(
        r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
        post)
    for link in links:
        post = post.replace(link[0], "LINK_TOKEN")
    return post

def bert_embed_document(post):
    #return bert_client.encode([doc_to_bert_interpertable(post)])[0]
    return model.encode(mask_links(post))

user_count = 0
save_every_n = 500
saved_files = 0
res_map = {}
file_map = {}

for index, row in source_frame.iterrows():
    print(user_count)
    user_count += 1
    if user_count > 0 and (user_count % save_every_n == 0):
        print("Saving last " + str(save_every_n) + " users...")
        pkl.dump(res_map, open(res_path + "_" + str(saved_files) + ".pkl" , 'wb'))
        res_map = {}
        saved_files += 1
    file_map[index] = str(res_path + "_" + str(saved_files) + ".pkl")
    try:
        for doc_tuple in row['documents']:
            try:
                pass
                res_map[doc_tuple[0]] = bert_embed_document(doc_tuple[1])
            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
        continue

pkl.dump(res_map, open(res_path + "_" + str(saved_files) + ".pkl" , 'wb'))
pkl.dump(file_map, open("data/twitter_dataset/embedding_file_paths.pkl" , 'wb'))