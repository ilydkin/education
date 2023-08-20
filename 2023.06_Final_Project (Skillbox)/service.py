import pandas as pd
import dill
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

with open('models/pipe1.pkl','rb') as file:
    pipe_1 = dill.load(file)

with open('models/pipe2.pkl','rb') as file:
    pipe_2 = dill.load(file)


class Request (BaseModel):
    session_id:Optional[str]
    client_id: Optional[str]
    visit_date: Optional[str]
    visit_time: Optional[str]
    visit_number: Optional[str]
    utm_source: Optional[str]
    utm_medium: Optional[str]
    utm_campaign: Optional[str]
    utm_adcontent: Optional[str]
    utm_keyword: Optional[str]
    device_category: Optional[str]
    device_os: Optional[str]
    device_brand: Optional[str]
    device_model: Optional[str]
    device_screen_resolution: Optional[str]
    device_browser: Optional[str]
    geo_country: Optional[str]
    geo_city: Optional[str]
    hit_page_path:Optional[str]


class Prediction(BaseModel):
    client_id: Optional[str]
    visit_date: Optional[str]
    visit_time: Optional[str]
    prediction: int

@app.get('/status')
def status():
    return 'connection established'

@app.get('/model_1')
def model_1():
    return pipe_1['metadata']

@app.get('/model_2')
def model_2():
    return pipe_2['metadata']

@app.post('/predict_with_model_1', response_model=Prediction)
def predict_with_model_1(request: Request):

    input_data = request.dict()
    filtered_columns = ['utm_source',
                        'utm_medium',
                        'utm_campaign',
                        'utm_adcontent',
                        'device_category',
                        'device_brand',
                        'device_screen_resolution',
                        'device_browser',
                        'geo_country',
                        'geo_city']

    df = pd.DataFrame.from_dict(input_data, orient='index').T
    df = df[filtered_columns]
    y = pipe_1['model'].predict(df)

    return {'client_id': request.client_id,
            'visit_date': request.visit_date,
            'visit_time': request.visit_time,
            'prediction': y[0]
            }

@app.post('/predict_with_model_2', response_model=Prediction)
def predict_with_model_2(request: Request):

    input_data = request.dict()
    filtered_columns = ['device_screen_resolution',
                        'utm_adcontent',
                        'geo_city',
                        'utm_campaign',
                        'hit_page_path']

    df = pd.DataFrame.from_dict(input_data, orient='index').T
    df = df[filtered_columns]
    y = pipe_2['model'].predict(df)

    return {'client_id': request.client_id,
            'visit_date': request.visit_date,
            'visit_time': request.visit_time,
            'prediction': y[0]
            }

if __name__ == '__main__':
    uvicorn.run('service:app', reload=True)

