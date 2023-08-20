import pandas as pd
import dill
import tzlocal

from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler(timezone=tzlocal.get_localzone())

with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)

df = pd.read_csv('model/data/homework.csv')


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(n=5)
    data['predicted_price_cat'] = model['model'].predict(data)
    print(data[['id','price', 'predicted_price_cat']])


if __name__ == '__main__':
    sched.start()