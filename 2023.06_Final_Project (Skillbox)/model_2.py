import pandas as pd
import datetime
import joblib
import dill

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.utils import resample
from loguru import logger


sessions_path = 'data/ga_sessions.pkl'
hits_path = 'data/ga_hits.pkl'
logs_path = 'data/logs/info.log'
df_path = 'data/clean.csv'

input_columns = ['event_action', 'device_screen_resolution', 'utm_adcontent',
                 'hit_page_path', 'geo_city', 'utm_campaign']


def target_to_Boolean(df):
    target_values = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                     'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                     'sub_submit_success', 'sub_car_request_submit_click']

    y = df['event_action'].apply(lambda x: 1 if x in target_values else 0)
    return y


def features_generator_2(X):
    from six.moves.urllib.parse import urlparse, parse_qs
    features_to_generate = ['utm_content_initial', 'path', 'utm_referrer_initial', 'fbclid',
                            'utm_campaign_initial', 'utm_term_initial']
    for feat in X.columns:
        X.loc[:, feat] = X[feat].fillna(X[feat].mode()[0])
    for elem in features_to_generate:
        if elem == 'path':
            X.loc[:, 'path'] = [str(urlparse(url).path) for url in X['hit_page_path']]
        else:
            X.loc[:, elem] = [str(parse_qs(urlparse(url).query).get(elem, [0])[0]) for url in X['hit_page_path']]
    X.drop('hit_page_path', axis=1, inplace=True)
    return X


def training_sample(df):
    sample = df.sample(n=10000000, random_state=34, ignore_index=True).drop_duplicates()
    y_ = target_to_Boolean(sample)
    majority = sample[y_ == 0]
    minority = sample[y_ == 1]
    majority_reduced = resample(majority, n_samples=len(minority), random_state=34)
    df_reduced = pd.concat([majority_reduced, minority])
    X = df_reduced.drop('event_action', axis=1)
    y = target_to_Boolean(df_reduced)
    X_train, _, y_train, __ = train_test_split(X, y, test_size=0.1, random_state=34)
    logger.info('Training complete')
    return X_train, y_train


def evaluation_sample(df):
    sample = df.sample(n=12000000, random_state=34, ignore_index=True).drop_duplicates()
    X = sample.drop('event_action', axis=1)
    y = target_to_Boolean(sample)
    _, X_test, __, y_test = train_test_split(X, y, test_size=0.1, random_state=34)
    logger.info('Evaluation complete')
    return X_test, y_test


def genuine_data():
    sessions = joblib.load(sessions_path)
    hits = joblib.load(hits_path)
    df = sessions.merge(hits, how='outer', on='session_id', copy=False).dropna(subset=['event_action'])
    return df


def main():
    logger.add(logs_path, level='INFO')

    # df = genuine_data()[input_columns] #if your memo allows you
    df = pd.read_csv(df_path, usecols=input_columns)  # from merged and cleaned file

    model = LogisticRegression(C=0.6)
    X_columns = df.drop('event_action', axis=1).columns

    X_transformer = Pipeline(steps=[
        ('features_generator', FunctionTransformer(features_generator_2)),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('X', X_transformer, X_columns)])
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)])

    X_train, y_train = training_sample(df)
    pipe.fit(X_train, y_train)

    X_test, y_test = evaluation_sample(df)
    y_predict = pipe.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_predict)

    logger.info(f'roc_auc on genuine data for {model} is: {roc_auc:.4}')

    with open('models/pipe2.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'target action prediction',
                'author': 'Ivan Lydkin',
                'LinkedIn': 'in/ivan-lydkin/',
                'version': 2.0,
                'date': datetime.datetime.now(),
                'type': type(model).__name__,
                'roc_auc': roc_auc,
                'hyperparameters': model.get_params()
            }
        }, file)


if __name__ == '__main__':
    main()
