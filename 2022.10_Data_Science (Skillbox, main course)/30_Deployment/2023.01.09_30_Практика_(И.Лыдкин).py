import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(f):
    df = f.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def fix_outliers(f):
    df = f.copy()
    data = df['year']

    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

    df.loc[:, 'year'] = df['year'].apply(lambda x: round(boundaries[0]) if x < boundaries[0] else (round(boundaries[1]) if x > boundaries[1] else x))
    return df


def new_features(f):
    df = f.copy()
    df.loc[:, 'short_model'] = df['model'].apply(lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df


def main():
    print('\n Practice #30 performed with tender loving care by Ivan Lydkin')

    df = pd.read_csv('30.6 homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('fix_outliers', FunctionTransformer(fix_outliers)),
        ('new_features', FunctionTransformer(new_features))
    ])

    transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector (dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include = object))
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('transformer', transformer),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, '30_practice.pkl')


if __name__ == '__main__':
    main()