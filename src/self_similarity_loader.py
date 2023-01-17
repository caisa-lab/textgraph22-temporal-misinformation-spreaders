import json 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

path = "data/stored_user_embeddings/avg/"
filename = "avg_user_embeddings_delta_30.json"

with open(path+filename) as f:
    data = f.read()

json_load = json.loads(data)

users = list(json_load.keys())
avg_cos_dict = {}

for user in users:
    embeddings = np.asarray(json_load[user])
    cos_mat = cosine_similarity(embeddings, embeddings)
    triu_cos = np.triu(cos_mat, k=1)
    nz_triu = triu_cos[np.where(triu_cos!=0)]
    avg_cos = np.mean(nz_triu)
    avg_cos_dict[user] = avg_cos


    

