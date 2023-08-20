import requests


length = 40
text = 'make some noise and epic lyrics'
# url = 'http://127.0.0.1:8000/predict'
url = 'http://84.201.137.60:8000/predict'
# url = 'http://84.201.137.60:8000/'


resp = requests.get(url, params={'text': text, 'length': length})
print(resp.status_code)
print(resp.content.decode('utf-8'))