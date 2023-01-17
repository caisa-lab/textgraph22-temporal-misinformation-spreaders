from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from classification.classification_baseline import perform_training_user_split
import datetime
from xgboost import XGBClassifier
import sys

SPLIT = "USER"

def exec_baseline_user_split(feature_calc, mode, run_id):
    models = [LinearSVC(max_iter=10000), SVC(max_iter=10000),
              RandomForestClassifier(n_estimators=300), LogisticRegression(max_iter=10000),
              XGBClassifier()]
    data_path = "data/core_dataset/reddit_corpus_final_balanced.gzip"
    split = (datetime.date(2020, 1, 1), datetime.date(2021, 4, 25))
    perform_training_user_split(data_path, split, models, split_type='user', fold_index=0, folds=4,
                                feature_calc=feature_calc, mode=mode, run_id=run_id)


def exec_baseline_time_split(feature_calc, mode, run_id):
    models = [LinearSVC(max_iter=10000), SVC(max_iter=10000),
              RandomForestClassifier(n_estimators=300), LogisticRegression(max_iter=10000),
              XGBClassifier()]
    data_path = "data/core_dataset/reddit_corpus_final_balanced.gzip"
    time_split = [(datetime.date(2020, 1, 1), datetime.date(2020, 8, 28)),
                  (datetime.date(2020, 8, 28), datetime.date(2020, 12, 26))
        ,(datetime.date(2020, 12, 26), datetime.date(2021, 4, 25))]
    perform_training_user_split(data_path, time_split, models, split_type='time', fold_index=-1, feature_calc=feature_calc,
                                mode=mode, run_id=run_id)


if __name__ == '__main__':
    feat_input = int(sys.argv[1])
    run_id = sys.argv[2]
    print("Running in feature mode: " + str(feat_input))
    print(run_id)
    if SPLIT == "USER":
        if feat_input == 0:
            exec_baseline_user_split("ngrams", "porter_stemmer", run_id)
        elif feat_input == 1:
            exec_baseline_user_split("ngrams", "wordnet_lemmatizer", run_id)
        elif feat_input == 2:
            exec_baseline_user_split("bert", "None", run_id)
        elif feat_input == 3:
            exec_baseline_user_split("char_ngrams_wb", "None", run_id)
        elif feat_input == 4:
            exec_baseline_user_split("surface", "None", run_id)
    elif SPLIT == "TIME":
        if feat_input == 0:
            exec_baseline_time_split("ngrams", "porter_stemmer", run_id)
        elif feat_input == 1:
            exec_baseline_time_split("ngrams", "wordnet_lemmatizer", run_id)
        elif feat_input == 2:
            exec_baseline_time_split("bert", "None", run_id)
        elif feat_input == 3:
            exec_baseline_time_split("char_ngrams_wb", "None", run_id)
        elif feat_input == 4:
            exec_baseline_time_split("surface", "None", run_id)"""
