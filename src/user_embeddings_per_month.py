from classification.feature_computing import *
import os

import pickle as pkl
from tqdm import tqdm
import glob
from constants import *

def sentence_embeddings_model(bert_model, num_features):
    model = SentenceTransformer(bert_model).to(DEVICE)
    
    return model


if __name__ == '__main__':
    
    model = sentence_embeddings_model('all-mpnet-base-v2', 768)
    
    embeddings_dir = '/app/home/plepi/temporal-misinformation-spreaders/data/twitter_dataset/user_vocabs_per_month'
    files = glob.glob(os.path.join(embeddings_dir, '*.txt'))
    files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

    for i, file in tqdm(enumerate(files), desc="File Processing Progress"):
        user_texts = {}
        embeddings = {'desc': 'Bert Embeddings per months'}

        with open(file, 'r') as f:
            for j, line in enumerate(f):
                temp = line.split('\t')
                assert len(temp) == 2, (temp, file, j)
                user = temp[0]
                text = temp[1]
                
                texts = user_texts.get(user, [])
                texts.append(text.strip())
                user_texts[user] = texts
        
        for user_id, texts in tqdm(user_texts.items(), desc="Embedding Progress"):
            texts = [process_tweet(text) for text in texts]
            output = model.encode(texts)
            embeddings[user_id] = torch.tensor(np.mean(output, axis=0))

        pkl.dump(embeddings, open(f'../data/twitter_dataset/bert_embeddings/user_embeddings_{i}.pkl', 'wb'))
