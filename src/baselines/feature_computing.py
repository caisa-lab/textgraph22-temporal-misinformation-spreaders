from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from utils.train_utils import process_tweet
import numpy as np
import pickle as pkl
import nltk as nlp
import string
import spacy
import random
from spacymoji import Emoji
from nltk import tokenize
from nltk.corpus import stopwords
import torch 
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from nltk import tokenize
from utils.train_utils import get_embeddings_dict_from_path
import abc
import glob
import math
import logging
import gzip
import re
import pandas as pd
from sklearn.preprocessing import normalize
import os

"""
This class bundles the different methodologies for feature vector calculation. 
"""
class Embedder:
    def __init__(self, embeddings_dir, embeddings_type='bert', dim=768, read_personality=True) -> None:
        self.users_embeddings = {}
        self.dim = dim
        if 'ezzeddine' in embeddings_type:
            typ = embeddings_type.split('_')[-1]
            filename = os.path.join(embeddings_dir[0], f'user_embedding_{typ}.p')
            self.users_embeddings = pkl.load(open(filename, 'rb'))
            for u, emb in self.users_embeddings.items():
                self.users_embeddings[u] = [emb]
                
        if embeddings_type == 'bert':
            files = glob.glob(os.path.join(embeddings_dir[0], '*.pkl'))
            files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

            for index, file in enumerate(files):
                temp_embeddings = pkl.load(open(file, 'rb'))
                for user_id, embedding in temp_embeddings.items():
                    if user_id != 'desc':
                        user_embedding = self.users_embeddings.get(user_id, [])
                        
                        if len(user_embedding) < index and len(user_embedding) > 0:
                            current = user_embedding[-1]
                        elif len(user_embedding) < index and len(user_embedding) == 0:
                            current = torch.rand(dim)
                        
                        while len(user_embedding) < index:
                            user_embedding.append(current)
                        
                        user_embedding.append(torch.tensor(embedding))
                        self.users_embeddings[user_id] = user_embedding
            
        if 'usr2vec' in embeddings_type:
            files = glob.glob(os.path.join(embeddings_dir[0], '*.txt'))
            files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

            for index, file in enumerate(files):
                with open(file) as f:
                    next(f)
                    for line in f:
                        values = line.split(' ')
                        user_id = values[0]
                        embedding = np.array(values[-200:]).astype(np.double)
                        user_embedding = self.users_embeddings.get(user_id, [])
                        #user_embedding.append(torch.tensor(embedding))
                        #self.users_embeddings[user_id] = user_embedding
                        if len(user_embedding) < index and len(user_embedding) > 0:
                            current = user_embedding[-1]
                        elif len(user_embedding) < index and len(user_embedding) == 0:
                            current = torch.rand(dim)
                    
                        while len(user_embedding) < index:
                            user_embedding.append(current)
                    
                        user_embedding.append(torch.tensor(embedding))
                        self.users_embeddings[user_id] = user_embedding     
            
            #for user, values in self.users_embeddings.items():
            #    self.users_embeddings[user] = [torch.stack(values).mean(axis=0)]
        
        if 'usr2vec' in embeddings_type and 'rand' in embeddings_type:
            for user, embedding in self.users_embeddings.items():
                self.users_embeddings[user]  = [torch.cat((embedding[0], torch.rand(83)))]
                    
                    
        if 'usr2vec' in embeddings_type and 'liwc' in embeddings_type:
            liwc_embeddings = {}
            liwc_frame = pd.read_pickle(os.path.join(embeddings_dir[1], 'new_static_LIWC_features.pkl'))
            for index, row in liwc_frame.iterrows():
                liwc_embeddings[index] = torch.tensor(row.values)
            
            if read_personality:
                personality_frame = pd.read_pickle(os.path.join(embeddings_dir[1], 'new_static_personality_features.pkl'))
                for index, row in personality_frame.iterrows():
                    v = liwc_embeddings[index]
                    liwc_embeddings[index] = [torch.cat((v, torch.tensor(row.values)))]
            
            for user, embedding in self.users_embeddings.items():
                if user in liwc_embeddings:
                    self.users_embeddings[user]  = [torch.cat((embedding[0], liwc_embeddings[user][0]))]
                else:
                    self.users_embeddings[user]  = [torch.cat((embedding[0], torch.rand(83)))]

        if embeddings_type == 'liwc':
            liwc_frame = pd.read_pickle(os.path.join(embeddings_dir[0], 'new_static_LIWC_features.pkl'))
            for index, row in liwc_frame.iterrows():
                self.users_embeddings[index] = torch.tensor(row.values)
            
            if read_personality:
                personality_frame = pd.read_pickle(os.path.join(embeddings_dir[0], 'new_static_personality_features.pkl'))
                for user, embedding in self.users_embeddings.items():
                    if user in personality_frame.index:
                        value = personality_frame.loc[user].values
                        self.users_embeddings[user] = [torch.cat((embedding, torch.tensor(value)))]
                    else:
                        self.users_embeddings[user] = [torch.cat((embedding, torch.rand(19)))]
            
        for user, values in self.users_embeddings.items():
            while len(values) < 16:
                values.append(torch.zeros(dim))
        
        self.test_users = []
        for user, values in self.users_embeddings.items():
            if torch.equal(values[-4].double(), torch.zeros(dim).double()) and torch.equal(values[-3].double(),torch.zeros(dim).double()) \
                and torch.equal(values[-2].double(), torch.zeros(dim).double()) and torch.equal(values[-1].double(), torch.zeros(dim).double()):
                pass
            else:
                self.test_users.append(user)    
    
    def embed_user(self, idx, time_bucket=None, mode='avg'):
        if time_bucket is None:
            if idx in self.users_embeddings:
                return torch.stack(self.users_embeddings[idx]).mean(axis=0)
            else:
                return torch.rand(self.dim)
        else:
            return self.users_embeddings[idx][time_bucket]
    
    def __del__(self):
        self.users_embeddings = {}


# class Embedder(abc.ABC):
#     @abc.abstractmethod
#     def embed_user(self, ids, mode='avg', timestamp_map={}):
#         pass


# class User2VecEmbedder(Embedder):
#     def __init__(self):
#         pass
    
#     def initialize(self, embeddings_dir, embeddings_file=None) -> None:
#         super().__init__()
#         if embeddings_file is None:
#             files = glob.glob(os.path.join(embeddings_dir, '*.txt'))
#             files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

#             self.users_embeddings = {}
#             for file in files:
#                 with open(file) as f:
#                     next(f)
#                     for line in f:
#                         values = line.split(' ')
#                         user_id = values[0]
#                         embedding = np.array(values[1:]).astype(np.double)
#                         user_embedding = self.users_embeddings.get(user_id, [])
#                         user_embedding.append(embedding)
#                         self.users_embeddings[user_id] = user_embedding
#         else:   
#             raise('Not implemented. Assumes the files are user_embeddings_0.txt until 15')
    
#     def embed_user(self, idx, mode='avg', timestamp_map={}, time_bucket=None):
#         if time_bucket is None:
#             return self.users_embeddings[idx]
#         else:
#             return self.users_embeddings[idx][time_bucket]
    
#     def __del__(self):
#         self.users_embeddings = {}
        

# class SavedPostBertEmbedder(Embedder):
    
#     def __init__(self):
#         pass

#     def initialize(self, post_embedding_file, no_file=False):
#         if not no_file:
#             self.file_name = post_embedding_file
#             if post_embedding_file.endswith('.gzip'):
#                 print("Loading embeddings file from gzip...")
#                 self.post_embedding_map = pkl.load(gzip.open(post_embedding_file, "rb"))
#             elif post_embedding_file.endswith('.txt'):
#                 print("Loading embeddings file from txt...")
#                 self.post_embedding_map = get_embeddings_dict_from_path(post_embedding_file)
#             else:
#                 print("Loading embeddings file from pkl...")
#                 self.post_embedding_map = pkl.load(open(post_embedding_file, "rb"))
#         else:
#             self.file_name = None
#             self.post_embedding_map = {}

#     def embed_document_by_id(self, document_id):
#         return self.post_embedding_map[document_id]

#     def normalize(self, v):
#         norm = np.linalg.norm(v, ord=2)
#         if norm == 0:
#             norm = np.finfo(v.dtype).eps
#         return np.array(v / norm)

#     def calc_weight_map(self, timestamp_map, d=10, beta=10, eps=4.5):
#         res = np.zeros(len(timestamp_map))
#         for index, ts in enumerate(timestamp_map.values()):
#             sorted_dists = sorted([abs((compare_ts - ts).total_seconds()) for compare_ts in timestamp_map.values()])
#             res[index] = np.average(sorted_dists[:d])

#         exp = self.normalize(res)
#         weights = np.array([1 + eps * math.exp(-beta * val) for val in exp])
#         weight_map = {}

#         for index, pid in enumerate(timestamp_map.keys()):
#             weight_map[pid] = weights[index]

#         return weight_map

#     def embed_user(self, ids, mode='avg', timestamp_map={}, time_bucket=None):
#         """
#         :param ids: Documents ids for a user
#         """
        
#         if mode == 'avg':
#             return np.average(np.array([self.post_embedding_map[str(pid)] for pid in list(ids)]), axis=0)
#         if mode == 'freq_weighted_avg':
#             weight_map = self.calc_weight_map(timestamp_map)
#             weight_sum = sum(weight_map.values())
#             val_sum = 0
#             for pid in ids:
#                 val_sum += weight_map[pid] * self.post_embedding_map[pid]
#             return val_sum / weight_sum
#         if mode == 'max':
#             return np.max(np.array([self.post_embedding_map[pid] for pid in list(ids)]), axis=0)
#         if mode == 'random':
#             temp = np.array([self.post_embedding_map[pid] for pid in list(ids)])
#             return np.random.randint()


class AbstractFeatureComputer(abc.ABC):
    """
    Abstract super class for the different types of feature computers.
    To be used in the abstract user prediction pipeline.
    """

    @abc.abstractmethod
    def transform(self, documents: [str]) -> []:
        """
        Calculate the feature vector of a user based on
        the list of the posts he produced,
        :param documents: The documents of the user to vectorize
        :return: The computed feature vector
        """
        pass


class NgramVectorizer(AbstractFeatureComputer):

    def __init__(self, analyzer='word', ngram_range=(1, 3), max_df=1.0, min_df=0.0001, stemmer=None):
        if stemmer is None:
            self.count_vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_df=max_df, min_df=min_df, token_pattern=None, stop_words=stopwords.words('english'))
        elif stemmer == 'porter_stemmer':
            sw_stemmer = PorterStemmer()
            stop = [sw_stemmer.stem(sw) for sw in stopwords.words('english')]
            self.count_vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                                    tokenizer=self.build_porter_tokenizer(), token_pattern=None, stop_words=stop)
        elif stemmer == 'wordnet_lemmatizer':
            sw_lemmatizer = WordNetLemmatizer()
            stop = [sw_lemmatizer.lemmatize(sw) for sw in stopwords.words('english')]
            self.count_vectorizer = CountVectorizer(analyzer=analyzer,ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                    tokenizer=self.build_wordnet_tokenizer(), token_pattern=None, stop_words=stop)    
        else:
            raise Exception('Invalid argument!')

    def build_porter_tokenizer(self):
        #Maybe change to TweetTokenizer
        print('Building porter stemmer')
        token_stemmer = PorterStemmer()
        return lambda doc: [token_stemmer.stem(token) for token in word_tokenize(process_tweet(doc.strip('\"')))]

    def build_wordnet_tokenizer(self):
        print('Building wordnet lemmatizer')
        lemmatizer = WordNetLemmatizer()
        return lambda doc: [lemmatizer.lemmatize(token) for token in word_tokenize(process_tweet(doc.strip('\"')))]

    def fit(self, corpus_docs: []):
        print('Fitting vectorizer...')
        self.count_vectorizer.fit(corpus_docs)
        print('Fitting completed')

    def transform(self, documents: [str]) -> []:
        if len(documents) == 0:
            return np.zeros(len(self.count_vectorizer.get_feature_names()))
        user_doc_vectors = self.count_vectorizer.transform(documents)
        user_doc_vectors = np.array(user_doc_vectors.toarray())
        avg = np.average(user_doc_vectors, axis=0)
        return avg


class TwitterTfIdfVectorizer(TfidfVectorizer):
    """
    Overrides the sklearn TfIdfVectorizer.
    Id adds a stemmer and a tweet tokenizer to the n_gram pipeline.
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()

    def stem(self, token: str):
        """
        Stem a given token using the PorterStemmer or the two hardcoded rules.

        :param token: Token to stem
        :return: Stemmed token
        """
        if token.startswith('@'):
            return "TWITTER_AT"
        if token.startswith('http'):
            return "LINK"
        else:
            return self.token_stemmer.stem(token)

    def build_tokenizer(self):
        """
        Overrides the build_tokenizer method.

        :return: The tokenizer lambda expression
        """
        return lambda doc: [self.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]


# class BertAsServiceEmbedder(AbstractFeatureComputer):
#     def __init__(self):
#         self.client = BertClient()
#         self.feature_dict = np.zeros(768)

#     def doc_to_bert_interpertable(self, doc):
#         """
#         Helper method for transferring a given document
#         into a bert-interpretable string, where senteces are
#         separated by '|||'
#         :param doc: The document to transform
#         :return: The bert-interpretable string
#         """
#         sentences = tokenize.sent_tokenize(doc)
#         res = ""
#         for index, sentence in enumerate(sentences):
#             res += sentence
#             if index < len(sentences) - 1:
#                 res += ' ||| '
#         return res

#     def transform(self, documents: [str]) -> []:
#         return np.average(self.client.encode([self.doc_to_bert_interpertable(doc) for doc in documents]), axis=0)


class TfIdfFeatureComputer(AbstractFeatureComputer):
    """
    Feature computer for the tf-idfs features.
    It utilises the sklearn tf-idf feature calculator
    """

    def __init__(self, train_data: []):
        """
        Constructor
        :param train_data: The documents that are used for training
        """
        self.feature_corpus = train_data
        self.feature_dict = []
        self.vectorizer = TwitterTfIdfVectorizer()
        self.build_feature_mapping(self.feature_corpus)
        logging.info(len(self.vectorizer.get_feature_names()))

    def build_feature_mapping(self, corpus: []):
        """
        Build a feature mapping based on the given corpus
        :param corpus: The documents to build the feaure mapping on
        """
        self.vectorizer.ngram_range = (1, 3)
        self.vectorizer.min_df = 80
        self.vectorizer.fit(corpus)
        self.feature_dict = [list() for i in range(len(self.vectorizer.vocabulary_))]

        for token, index in self.vectorizer.vocabulary_.items():
            self.feature_dict[index] = token

    def vectorize_data(self, data: []):
        """

        :param data:
        :return:
        """
        return [self.vectorizer.transform([document]).toarray()[0] for document in data]

    def transform(self, user_documents: []):
        """
        Calculate the average tf-idf feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated average tf-idf feature vector
        """
        if len(user_documents) == 0:
            return np.zeros(len(self.feature_dict))
        user_doc_vectors = self.vectorize_data(user_documents)
        user_doc_vectors = [np.array(vec) for vec in user_doc_vectors]
        avg = np.average(user_doc_vectors, axis=0)
        return avg

    def vectorize_bag_of_docs(self, data: [[]]):
        """
        Vectorize the data of multiple users
        :param data: The documents of the users as a list of lists
        :return: The calculatedfeature vectors in the same order
        """
        res_data = []
        for doc_set in data:
            user_doc_vectors = self.vectorize_data(doc_set)
            user_doc_vectors = [np.array(vec) for vec in user_doc_vectors]
            avg = np.average(user_doc_vectors, axis=0)
            res_data.append(avg)

        return res_data


class SurfaceFeatureComputer(AbstractFeatureComputer):
    """
    Feature computer for the abstract/surface features.
    Name those are:
    -The avg number of sentences
    -The avg number of emojis
    -The profanity ratio
    -The avg number of token
    -The avg number of ats
    -The avg number of links
    -The avg number of hashtags
    """

    def __init__(self):
        """
        Constructor
        """
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm')
        emoji = Emoji(self.nlp)
        self.nlp.add_pipe(emoji, first=True)
        self.word_emotion_dict = {}
        self.load_emotion_dict()
        print(self.word_emotion_dict)
        self.feature_dict = ["MEAN_SENT_AMOUNT", "EMOJI", "PROFANITY", "MIN_TOKEN_N", "MAX_TOKEN_N", "MIN_CHAR_N",
                             "MAX_CHAR_N",
                             "MEAN_TOKEN_N", "MEAN_CHAR_N", "RANGE_TOKEN_N", "RANGE_CHAR_N",
                             "STD_TOKEN_N", "STD_CHAR_N", "TTR", "LINKS", "CAPS",
                             "EMO_ANGER", "EMO_ANTICIPATION", "EMO_DISGUST", "EMO_FEAR", "EMO_JOY",
                             "EMO_NEGATIVE", "EMO_POSITIVE", "EMO_SADNESS", "EMO_SURPRISE", "EMO_TRUST"]

    def load_emotion_dict(self):
        path = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        emolex_df = pd.read_csv(path, names=["word", "emotion", "association"], sep='\t')
        for index, row in emolex_df.iterrows():
            if row['association'] == 0:
                continue
            if row['word'].lower() in self.word_emotion_dict.keys():
                self.word_emotion_dict[row['word'].lower()].append(row['emotion'])
            else:
                self.word_emotion_dict[row['word'].lower()] = [row['emotion']]

    def transform(self, documents: []):
        """
        Calculate the average surface feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated average surface feature vector
        """
        doc_tokens = [self.tokenize(doc) for doc in documents]

        vector = np.array(self.get_doc_based_averages(documents))
        vector = np.append(vector, [self.get_token_based_averages(doc_tokens)])

        return vector

    @staticmethod
    def regex_matches_full(regex, token):
        res = re.match(regex, token)
        if res is None:
            return False
        else:
            return len(res.group(0)) == len(token)

    def tokenize(self, doc: str):
        """
        Helper method for tokenizing a given post.
        :param doc: The post to tokenize
        :return: The resulting tokens
        """
        return [token for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]

    def get_token_based_averages(self, doc_tokens: [[]]):
        """
        Method for tracking all token based features.
        They are all tracked in the same loop for
        time efficiency reasons

        :param doc_tokens: The tokenized documents of a user
        :return: The token based feature vector
        """
        num_tokens_sum = 0
        token_amounts = []
        char_amounts = []
        num_ats_sum = 0
        num_hashtags_sum = 0
        num_links_sum = 0
        emotion_counter = {
            'anger': 0,
            'anticipation': 0,
            'disgust': 0,
            'fear': 0,
            'joy': 0,
            'negative': 0,
            'positive': 0,
            'sadness': 0,
            'surprise': 0,
            'trust': 0
        }
        tokens_full = []
        caps_token = 0
        for tokenized in doc_tokens:
            tokenized = [token for token in tokenized if token not in string.punctuation]
            tokens_full.extend([self.token_stemmer.stem(token) for token in tokenized])
            num_tokens_sum = num_tokens_sum + len(tokenized)
            token_amounts.append(len(tokenized))
            n_chars = 0
            for token in tokenized:
                try:
                    for emo in self.word_emotion_dict[token.lower()]:
                        emotion_counter[emo] += 1
                except Exception as e:
                    pass
                if SurfaceFeatureComputer.regex_matches_full(r'[A-Z]+', token):
                    caps_token += 1
                n_chars += len(token)
                if token.strip().startswith('@'):
                    num_ats_sum = num_ats_sum + 1
                if token.strip().startswith('http'):
                    num_links_sum = num_links_sum + 1
                if token.strip().startswith('#'):
                    num_hashtags_sum = num_hashtags_sum + 1
            char_amounts.append(n_chars)

        types = nlp.Counter(tokens_full)

        print(emotion_counter)

        return [min(token_amounts), max(token_amounts), min(char_amounts), max(char_amounts),
                np.mean(token_amounts), np.mean(char_amounts),
                max(token_amounts) - min(token_amounts), max(char_amounts) - min(char_amounts),
                np.std(token_amounts), np.std(char_amounts),
                (len(types) / len(tokens_full)),
                # num_ats_sum / len(doc_tokens),
                # num_hashtags_sum / len(doc_tokens),
                num_links_sum / len(doc_tokens),
                caps_token / sum(token_amounts),
                emotion_counter['anger']/sum(token_amounts),
                emotion_counter['anticipation']/sum(token_amounts),
                emotion_counter['disgust']/sum(token_amounts),
                emotion_counter['fear']/sum(token_amounts),
                emotion_counter['joy']/sum(token_amounts),
                emotion_counter['negative']/sum(token_amounts),
                emotion_counter['positive']/sum(token_amounts),
                emotion_counter['sadness']/sum(token_amounts),
                emotion_counter['surprise']/sum(token_amounts),
                emotion_counter['trust']/sum(token_amounts)
                ]

    def get_doc_based_averages(self, documents: []):
        """
        Method for tracking all sentence based features.
        They are all tracked in the same loop for
        time efficiency reasons

        :param documents: The posts of a user
        :return: The sentence based feature vector
        """
        sent_sum = 0
        emoji_sum = 0
        profanity_sum = 0
        politness_sum = 0
        for doc in documents:
            sent_sum += len(tokenize.sent_tokenize(doc))

            try:
                scanned = self.nlp(doc)
                emoji_sum = emoji_sum + len(scanned._.emoji)
            except ValueError:
                continue

            # try:
            #     if profanity_check.predict([doc])[0] == 1:
            #         profanity_sum = profanity_sum + 1
            # except RuntimeError:
            #     continue

        return [sent_sum / len(documents),
                emoji_sum / len(documents),
                profanity_sum / len(documents)]


class WordToVecTopicVectorizer(AbstractFeatureComputer):
    """
    Feature computer for the word2vec-cluster features.
    """

    def __init__(self):
        """
        Constructor
        """
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()
        self.word_to_vec_model = None
        self.cluster_mapping = {}
        self.cluster_amount = -1
        self.feature_dict = []

    def tokenize(self, doc: str):
        """
        Helper method for tokenizing a given document
        :param doc: The document to tokenize
        :return: The extracted tokens
        """
        return [self.token_stemmer.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]

    def tweet_to_tokenized_sentences(self, doc) -> [[]]:
        """
        Transfer a given post to a list of list with the tokens of each sentence
        in the post.

        :param doc: The post to tokenize
        :return: The split sentences as their a list of their tokens
        """
        res = []

        if not isinstance(doc, str):
            logging.warning("Unexpected type - skipping document")
            logging.warning(doc)
            return res

        for sent in tokenize.sent_tokenize(doc):
            sent_res = []
            sent_tokens = self.tokenize(sent)
            for i, token in enumerate(sent_tokens):
                if token.startswith('@'):
                    sent_tokens[i] = "TWITTER_AT"
                if token.startswith('http'):
                    sent_tokens[i] = "LINK"
            sent_res.extend(sent_tokens)
            res.append(sent_res)
        return res

    def fit(self, all_docs: []):
        """
        Fit the word2vec-cluster vectorizer.
        :param all_docs: All the documents for training
        """
        sentences = []

        for doc in all_docs:
            sentences.extend(self.tweet_to_tokenized_sentences(doc))

        self.word_to_vec_model = Word2Vec(sentences, min_count=1, sg=1)
        # Build word2vec dictionary
        w2v_indices = {word: self.word_to_vec_model.wv[word] for word in self.word_to_vec_model.wv.vocab}
        clustering_data = [*w2v_indices.values()]

        self.cluster_amount = 1000
        # Cluster word2vec dictionary
        kclusterer = MiniBatchKMeans(self.cluster_amount, max_iter=100, init_size=3000)
        logging.info("Clustering dictionary...")
        prediction_vector = kclusterer.fit_predict(clustering_data)

        index = 0
        for word, vec in w2v_indices.items():
            self.cluster_mapping[word] = prediction_vector[index]
            index += 1

        self.feature_dict = [list() for i in range(self.cluster_amount)]
        for word, cluster in self.cluster_mapping.items():
            self.feature_dict[cluster].append(word)

    def transform(self, user_documents: []):
        """
        Calculate the word2vec-cluster feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated word2vec-cluster feature vector
        """
        res_vector = np.zeros(self.cluster_amount)

        sent_tokenized = [self.tweet_to_tokenized_sentences(doc) for doc in user_documents]
        token_count = 0

        for tokenized in sent_tokenized:
            for tokens in tokenized:
                token_count += len(tokens)
                for token in tokens:
                    if token in self.cluster_mapping.keys():
                        res_vector[self.cluster_mapping[token]] += 1

        if token_count == 0:
            return res_vector
        else:
            return res_vector / token_count


class BagOfWordsVectorizer(AbstractFeatureComputer):
    """
    Feature computer for the unigram bag of words features.
    """

    def __init__(self):
        """
        Constructor
        """
        self.word_mapping = {}
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()
        self.feature_dict = []

    def tokenize(self, docs: []):
        """
        Helper method for tokenizing a given list of documents.
        :param docs: The list of documents to tokenize
        :return: The extracted tokens for each document
        """
        res = []
        for doc in docs:
            if not isinstance(doc, str):
                logging.warn("Unexpected type - skipping document")
                continue
            tokens = [self.token_stemmer.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]
            for index, token in enumerate(tokens):
                if token.startswith('@'):
                    tokens[index] = "TWITTER_AT"
                if token.startswith('http'):
                    tokens[index] = "LINK"
            res.append(tokens)
        return res

    def fit(self, documents_of_users: [[]]):
        """
        Fit the unigrams bag of words feautre computer
        to the given training data.
        :param documents_of_users: training data in the shape
        of a list of lists with each users documents
        """
        token_user_counter = {}

        index = 0
        for user_docs in documents_of_users:
            index += 1
            tokenized = self.tokenize(user_docs)

            for doc_tokens in tokenized:
                seen_tokens = []
                for token in doc_tokens:
                    if token not in seen_tokens:
                        seen_tokens.append(token)
                        if token in token_user_counter:
                            token_user_counter[token] = token_user_counter[token] + 1
                        else:
                            token_user_counter[token] = 1

        user_amount = len(documents_of_users)
        min_count = user_amount * 0.01

        logging.info("Min count:" + str(min_count))

        index = 0
        for token, count in token_user_counter.items():
            if count > min_count:
                self.word_mapping[token] = index
                index = index + 1

        self.feature_dict = [-1] * index
        for word, index in self.word_mapping.items():
            self.feature_dict[index] = word

    def transform(self, user_documents: []):
        """
        Calculate the unigrams feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated unigrams feature vector
        """
        bow_vector = np.zeros(len(self.word_mapping.keys()))
        token_sum = 0
        for tokenized in self.tokenize(user_documents):
            token_sum += len(tokenized)
            for token in tokenized:
                if token in self.word_mapping.keys():
                    bow_vector[self.word_mapping[token]] = bow_vector[self.word_mapping[token]] + 1

        # Normalize
        if len(user_documents) > 0:
            bow_vector = bow_vector / len(user_documents)
        else:
            bow_vector = np.zeros(len(self.word_mapping.keys()))
        return bow_vector
