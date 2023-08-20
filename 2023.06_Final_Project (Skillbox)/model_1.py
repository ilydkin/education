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

input_columns= ['event_action', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                'device_category', 'device_brand', 'device_screen_resolution', 'device_browser',
                'geo_country', 'geo_city']

def target_to_Boolean (df):
    target_values = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                     'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                     'sub_submit_success', 'sub_car_request_submit_click']

    y = df['event_action'].apply(lambda x: 1 if x in target_values else 0)
    return y


def features_generator_1(X):
    organic_values = ['organic', 'referral', '(none)']
    SMM_values = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                  'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    for feat in X.columns:
        X.loc[:, feat] = X[feat].fillna(X[feat].mode()[0])
    X.loc[:, 'organic_traffic'] = X['utm_medium'].apply(lambda x: 1 if x in organic_values else 0)
    X.loc[:, 'SMM'] = X['utm_source'].apply(lambda x: 1 if x in SMM_values else 0)
    return X


def training_sample (df):
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


def evaluation_sample (df):
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
    df = pd.read_csv(df_path, usecols=input_columns) # from merged and cleaned file

    model = LogisticRegression()
    X_columns = df.drop('event_action', axis=1).columns

    X_transformer = Pipeline(steps=[
        ('features_generator', FunctionTransformer(features_generator_1)),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('X', X_transformer, X_columns)])
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)])

    X_train, y_train = training_sample(df)
    pipe.fit (X_train,y_train)

    X_test, y_test = evaluation_sample (df)
    y_predict = pipe.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_predict)

    logger.info(f'roc_auc on genuine data for {model} is: {roc_auc:.4}')

    with open('models/pipe1.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Target Action Prediction',
                'author': 'Ivan Lydkin',
                'LinkedIn': 'in/ivan-lydkin/',
                'version': 1.0,
                'date': datetime.datetime.now(),
                'type': type(model).__name__,
                'roc_auc':  roc_auc,
                'hyperparameters': model.get_params()
            }
        }, file)


if __name__ == '__main__':
    main()
