import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator

path = os.path.join(os.path.expanduser('~'), 'airflow_hw')
sys.path.insert(0, path)

from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2023, 1, 25),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=3),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction_by_ivan_lydkin',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag=dag
    )

    prediction = PythonOperator(
        task_id= 'prediction',
        python_callable=predict,
        dag=dag
    )

    pipeline >> prediction

