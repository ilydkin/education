import json
import dill
import pandas as pd


def main():
    with open('model/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    with open('jsons/7313922964.json') as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        print(f'{form["id"]}: {y[0]}')


if __name__ == '__main__':
    main()