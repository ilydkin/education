import os
import json
import dill
import pandas as pd
import logging
from datetime import datetime


def predict():

    path = os.path.join(os.path.expanduser('~'), 'airflow_hw')
    model_name = os.listdir(f'{os.path.expanduser("~")}/airflow_hw/data/models')[0]

    with open(f'{os.path.join(path,"data", "models", model_name)}', 'rb') as file:
        model = dill.load(file)

    testing_list = os.listdir(f'{os.path.expanduser("~")}/airflow_hw/data/test')
    predictions = {}

    for test in testing_list:
        with open(f'{os.path.join(path,"data", "test", test)}') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            predictions[form["id"]] = y[0]

    df_pred=pd.DataFrame.from_dict(data=predictions, orient='index',  columns= ['price_category'])
    logging.info(f'DataFrame with the resulting predictions: {df_pred}')

    df_filename = f'{os.path.join(path,"data","predictions")}/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'

    df_pred.to_csv(df_filename, index_label='id')


if __name__ == '__main__':
    predict()
