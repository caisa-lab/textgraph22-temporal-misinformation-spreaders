import pandas as pd
from classification.feature_computing import SavedPostBertEmbedder
from tqdm import tqdm

test_file = "data/stored_embeddings/doc_embeddings.txt"

embedder = SavedPostBertEmbedder(test_file)
#print(embedder.post_embedding_map)

print("Embedder loading completed!")

source_frame = pd.read_pickle('data/core_dataset/core_dataset.gzip', compression='gzip')

for index, row in tqdm(source_frame.iterrows()):
    embedder.embed_user([doc[0] for doc in row['documents']])

print('Completed!')
