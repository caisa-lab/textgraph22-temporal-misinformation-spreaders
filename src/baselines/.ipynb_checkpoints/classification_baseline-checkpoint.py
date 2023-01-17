from data_collection.reddit_user_dataset import RedditUserDataset
from .feature_computing import BagOfWordsVectorizer
from .feature_computing import TfIdfFeatureComputer
from .feature_computing import SurfaceFeatureComputer
from .feature_computing import NgramVectorizer
from .feature_computing import SavedPostBertEmbedder
from .cross_validation_training import evaluate_model_coefficients
import numpy as np
import datetime
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import re

subreddits = ['r/RadicalFeminismUSA', 'r/GlobalClimateChange', 'r/TrueAntiVaccination', 'r/GunsAreCool',
              'r/LockdownCriticalLeft', 'r/antivax', 'r/uspolitics', 'r/Feminism', 'r/NoNewNormal', 'r/DebateVaccines',
              'r/democrats', 'r/RepublicanValues', 'r/Liberal', 'r/CoronavirusCanada', 'r/CoronavirusUS',
              'r/guncontrol', 'r/DebateVaccine', 'r/antifeminists', 'r/gunpolitics', 'r/NoLockdownsNoMasks',
              'r/Impeach_Trump', 'r/masculism', 'r/feminisms', 'r/AntiVaxxers', 'r/feminismformen', 'r/RadicalFeminism',
              'r/conservatives', 'r/JoeBiden', 'r/COVID19', 'r/CoronavirusUK', 'r/climateskeptics', 'r/CovidVaccinated',
              'r/EndTheLockdowns', 'r/Egalitarianism', 'r/Coronavirus', 'r/MRActivism', 'r/politics', 'r/climate',
              'r/CoronavirusRecession', 'r/prochoice', 'r/Firearms', 'r/Abortiondebate', 'r/offmychest', 'r/Masks4All',
              'r/abortion', 'r/GunResearch', 'r/AskProchoice', 'r/MensRights', 'r/prolife', 'r/progun',
              'r/GenderCritical', 'r/GunDebates', 'r/liberalgunowners', 'r/COVID19positive', 'r/Republican',
              'r/Conservative', 'r/VACCINES', 'r/LockdownSkepticism', 'r/climatechange', 'r/insaneprolife',
              'r/ConservativesOnly', 'r/5GDebate', 'r/vaxxhappened']


def normalize(features):
    print((len(features), len(features[0])))
    input = np.zeros((len(features), len(features[0])))  # Stack
    for index, feat in enumerate(features):
        print(feat)
        input[index] = feat
    input_mean = np.mean(input)
    input_std = np.std(input)
    input = input - input_mean
    input = input / input_std
    res = []
    for index, feature in enumerate(features):
        res.append(input[index])
    return res


def chunk_data(to_chunk, amount):
    chunk_size = len(to_chunk) // amount
    chunk_size = max(1, chunk_size)
    return list((to_chunk[i:i + chunk_size] for i in range(0, len(to_chunk), chunk_size)))[:amount]


def fold_data(chunked_data: [[]], fold_index: int):
    train_data = []
    val_data = []
    test_data = []

    for index in range(0, len(chunked_data)):
        if index == fold_index:
            test_data.extend(chunked_data[index])
        elif index == (fold_index + 1) % len(chunked_data):
            val_data.extend(chunked_data[index])
        else:
            train_data.extend(chunked_data[index])

    return train_data, val_data, test_data


def norm(x):
    if sum(x) == 0:
        return x
    return x / np.linalg.norm(x)


def get_embeddings(frame, embedding_file_path, embed_mode='avg'):
    group_by = frame.groupby(by='embedding_file')
    embedding_dict = {}
    for group_tuple in group_by:
        embedder = None
        embedder = SavedPostBertEmbedder(os.path.join(embedding_file_path, group_tuple[0]))
        for index, row in group_tuple[1].iterrows():
            if len([doc[0] for doc in row['documents']]) == 0:
                embedding_dict[index] = np.zeros(768)
                continue
            try:
                timestamp_dict = {}
                for doc in row['documents']:
                    post_date = doc[2]
                    if isinstance(post_date, str):
                        post_date = datetime.strptime(doc[2], '%Y-%m-%d %H:%M:%S')
                    timestamp_dict[doc[0]] = post_date
                embedding_dict[index] = embedder.embed_user(
                    [doc[0] for doc in row['documents']], mode=embed_mode,
                    timestamp_map=timestamp_dict)
            except Exception as e:
                print("Exception while embedding user " + str(index))
                print(e)
    embedder = None
    return embedding_dict


def contains_link(doc):
    return len(re.findall(
        r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
        doc)) > 0


def count_links(docs):
    sum = 0
    for post in docs:
        links = re.findall(
            r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))',
            post[1])
        sum += len(links)
    return sum


def count_subreddits(docs):
    res_map = np.zeros(len(subreddits))

    for doc in docs:
        for index, sr in enumerate(subreddits):
            if doc[3] == sr:
                res_map[index] += 1

    print(res_map)
    return res_map


def calc_prediction_map(model, train_features, train_ids, val_features, val_ids, test_features, test_ids):
    train_map = {}
    val_map = {}
    test_map = {}
    for counter, uid in enumerate(train_ids):
        train_map[uid] = [0, 0]
        train_map[uid][model.predict([train_features[counter]])[0]] += 1

    for counter, uid in enumerate(val_ids):
        val_map[uid] = [0, 0]
        val_map[uid][model.predict([val_features[counter]])[0]] += 1

    for counter, uid in enumerate(test_ids):
        test_map[uid] = [0, 0]
        test_map[uid][model.predict([test_features[counter]])[0]] += 1

    return train_map, val_map, test_map


def perform_training_user_split(source_dataset_path, split_dates, models, split_type='user', folds=5, fold_index=0,
                                feature_calc='bert', mode=None, run_id="baseline", emb_map=None, embedding_file_path='data/stored_embeddings/'):
    source_dataset = RedditUserDataset.load_from_file(source_dataset_path, compression='gzip')
    user_ids = []
    for index, row in source_dataset.data_frame.iterrows():
        user_ids.append(row['user_id'])

    if split_type == 'user':
        chunked = chunk_data(user_ids, folds)
        train_ids, val_ids, test_ids = fold_data(chunked, fold_index)
        print(len(train_ids))
        print(len(val_ids))
        print(len(test_ids))

        # Split validation
        for id in train_ids:
            if id in val_ids:
                raise Exception("Invalid split!")
            if id in test_ids:
                raise Exception("Invalid split")

        for id in val_ids:
            if id in test_ids:
                raise Exception("Invalid split!")

        source_dataset.timeframed_documents(split_dates)

        train_set = source_dataset.filter_user_ids(train_ids, inplace=False)
        print(len(train_set.data_frame))
        val_set = source_dataset.filter_user_ids(val_ids, inplace=False)
        print(len(val_set.data_frame))
        test_set = source_dataset.filter_user_ids(test_ids, inplace=False)
        print(len(test_set.data_frame))
    if split_type == 'time':
        train_set = source_dataset.timeframed_documents(split_dates[0], inplace=False)
        print(len(train_set.data_frame))
        val_set = source_dataset.timeframed_documents(split_dates[1], inplace=False)
        print(len(val_set.data_frame))
        test_set = source_dataset.timeframed_documents(split_dates[2], inplace=False)
        print(len(test_set.data_frame))

    if feature_calc == 'unigrams':
        training_docs = []
        tr_ids = []
        val_docs = []
        v_ids = []
        test_docs = []
        t_ids = []
        for index, row in train_set.data_frame.iterrows():
            user_docs = [doc_tuple[1] for doc_tuple in row['documents']]
            training_docs.append(user_docs)
            tr_ids.append(index)

        for index, row in val_set.data_frame.iterrows():
            user_docs = [doc_tuple[1] for doc_tuple in row['documents']]
            val_docs.append(user_docs)
            v_ids.append(index)

        for index, row in test_set.data_frame.iterrows():
            user_docs = [doc_tuple[1] for doc_tuple in row['documents']]
            test_docs.append(user_docs)
            t_ids.append(index)

        vectorizer = BagOfWordsVectorizer()
        vectorizer.fit(training_docs)

        print("Training features...")
        training_features = [vectorizer.transform(user_docs) for user_docs in training_docs]
        print("Val features...")
        val_features = [vectorizer.transform(user_docs) for user_docs in val_docs]
        print("Test features...")
        test_features = [vectorizer.transform(user_docs) for user_docs in test_docs]
    if feature_calc == 'tf-idf':
        training_docs = []
        tr_ids = []
        fitting_docs = []
        val_docs = []
        v_ids = []
        test_docs = []
        t_ids = []
        for index, row in train_set.data_frame.iterrows():
            user_docs = [doc_tuple[1] for doc_tuple in row['documents']]
            training_docs.append(user_docs)
            fitting_docs.extend(user_docs)
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            user_docs = [doc_tuple[1] for doc_tuple in row['documents']]
            val_docs.append(user_docs)
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            user_docs = [doc_tuple[1] for doc_tuple in row['documents']]
            test_docs.append(user_docs)
            t_ids.append(index)

        vectorizer = TfIdfFeatureComputer(fitting_docs)

        print("Training features...")
        training_features = [vectorizer.transform(user_docs) for user_docs in training_docs]
        print("Val features...")
        val_features = [vectorizer.transform(user_docs) for user_docs in val_docs]
        print("Test features...")
        test_features = [vectorizer.transform(user_docs) for user_docs in test_docs]
    if feature_calc == 'bert':
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        train_embeddings = get_embeddings(train_set.data_frame, embedding_file_path, embed_mode='avg')
        val_embeddings = get_embeddings(val_set.data_frame, embedding_file_path, embed_mode='avg')
        test_embeddings = get_embeddings(test_set.data_frame, embedding_file_path, embed_mode='avg')
        for index, row in train_set.data_frame.iterrows():
            training_features.append(train_embeddings[index])
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append(val_embeddings[index])
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append(test_embeddings[index])
            t_ids.append(index)
    if feature_calc == 'num_links':
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        for index, row in train_set.data_frame.iterrows():
            training_features.append([count_links(row['documents'])])
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append([count_links(row['documents'])])
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append([count_links(row['documents'])])
            t_ids.append(index)

    if feature_calc == 'subreddits':
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        for index, row in train_set.data_frame.iterrows():
            training_features.append(count_subreddits(row['documents']))
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append(count_subreddits(row['documents']))
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append(count_subreddits(row['documents']))
            t_ids.append(index)
    if feature_calc == 'ngrams':
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        vectorizer = NgramVectorizer(stemmer=mode)
        fit_corpus = []
        for index, row in train_set.data_frame.iterrows():
            fit_corpus.extend([doc_tuple[1] for doc_tuple in row['documents']])
        vectorizer.fit(fit_corpus)
        num_features = len(vectorizer.count_vectorizer.get_feature_names())
        print('num_features: ' + str(num_features))
        for index, row in train_set.data_frame.iterrows():
            training_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            t_ids.append(index)
    if feature_calc == 'char_ngrams_wb':
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        vectorizer = NgramVectorizer(analyzer='char_wb', stemmer=mode)
        fit_corpus = []
        for index, row in train_set.data_frame.iterrows():
            fit_corpus.extend([doc_tuple[1] for doc_tuple in row['documents']])
        vectorizer.fit(fit_corpus)
        num_features = len(vectorizer.count_vectorizer.get_feature_names())
        print('num_features: ' + str(num_features))
        for index, row in train_set.data_frame.iterrows():
            training_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            t_ids.append(index)
    if feature_calc == 'surface':
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        vectorizer = SurfaceFeatureComputer()
        num_features = len(vectorizer.feature_dict)
        print('num_features: ' + str(num_features))
        for index, row in train_set.data_frame.iterrows():
            training_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append(vectorizer.transform([doc_tuple[1] for doc_tuple in row['documents']]))
            t_ids.append(index)
    if feature_calc == 'precalculated':
        print("Using precalculated features...")
        training_features = []
        tr_ids = []
        val_features = []
        v_ids = []
        test_features = []
        t_ids = []
        for index, row in train_set.data_frame.iterrows():
            training_features.append(emb_map[index])
            tr_ids.append(index)
        for index, row in val_set.data_frame.iterrows():
            val_features.append(emb_map[index])
            v_ids.append(index)
        for index, row in test_set.data_frame.iterrows():
            test_features.append(emb_map[index])
            t_ids.append(index)

    labels = []
    for index, row in source_dataset.data_frame.iterrows():
        labels.append(row['fake_news_spreader'])

    print("Fitting models...")

    for model in models:
        print('Training with model ' + str(model))
        model.fit(training_features, train_set.data_frame['fake_news_spreader'])
        TIMESTAMP = str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")
        res_map = {}
        run_id = run_id + "_" + str(model) + "_" + TIMESTAMP
        res_map['id'] = run_id
        res_map['timestamp'] = TIMESTAMP
        res_map['base_dataset'] = source_dataset_path
        res_map['type'] = str(model)
        res_map['timeframe'] = str(split_dates)
        res_map['user_fold_amount'] = folds
        res_map['user_fold_index'] = fold_index
        res_map['train_user_amount'] = len(train_set.data_frame['fake_news_spreader'])
        res_map['val_user_amount'] = len(val_set.data_frame['fake_news_spreader'])
        res_map['test_user_amount'] = len(test_set.data_frame['fake_news_spreader'])
        res_map['train_spreader_amount'] = sum(train_set.data_frame['fake_news_spreader'])
        res_map['val_spreader_amount'] = sum(val_set.data_frame['fake_news_spreader'])
        res_map['test_spreader_amount'] = sum(test_set.data_frame['fake_news_spreader'])

        train_score = model.score(training_features, train_set.data_frame['fake_news_spreader'])
        print("Train Accuray: " + str(train_score))
        print("Confusion Matrix:")
        res_map['train_acc'] = train_score
        print(confusion_matrix(train_set.data_frame['fake_news_spreader'],
                           [model.predict([vec])[0] for vec in training_features]))
        res_map['train_conf'] = str(confusion_matrix(train_set.data_frame['fake_news_spreader'],
                                                 [model.predict([vec])[0] for vec in training_features])).replace('\n',
                                                                                                                  '')

        val_score = model.score(val_features, val_set.data_frame['fake_news_spreader'])
        print("Val Accuracy: " + str(val_score))
        print("Confusion Matrix:")
        res_map['val_acc'] = val_score
        print(confusion_matrix(val_set.data_frame['fake_news_spreader'], [model.predict([vec])[0] for vec in val_features]))
        res_map['val_conf'] = str(confusion_matrix(val_set.data_frame['fake_news_spreader'],
                                               [model.predict([vec])[0] for vec in val_features])).replace('\n', '')

        test_score = model.score(test_features, test_set.data_frame['fake_news_spreader'])
        print("Test Accuracy: " + str(test_score))
        print("Confusion Matrix:")
        res_map['test_acc'] = test_score
        print(
            confusion_matrix(test_set.data_frame['fake_news_spreader'], [model.predict([vec])[0] for vec in test_features]))
        res_map['test_conf'] = str(confusion_matrix(test_set.data_frame['fake_news_spreader'],
                                                [model.predict([vec])[0] for vec in test_features])).replace('\n', '')

        #if 'LinearSVC' in str(model):
        #    bot, top = evaluate_model_coefficients('svm', model.coef_[0], vectorizer.count_vectorizer.get_feature_names())
        #    res_map['bottom_token_importances'] = str(bot)
        #    res_map['top_token_importances'] = str(top)
    
        res_map['features'] = feature_calc
        if feature_calc == 'ngrams':
            res_map['num_features'] = num_features

        train_preds, val_preds, test_preds = calc_prediction_map(model, training_features, tr_ids, val_features, v_ids,
                                                             test_features, t_ids)

        res_map['train_predictions'] = train_preds
        res_map['val_predictions'] = val_preds
        res_map['test_predictions'] = test_preds

        res_file = os.path.join("results/results_baseline", run_id + ".json")

        with open(res_file, mode='w') as f:
            f.write(json.dumps(res_map, indent=2))
