
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/hour.csv"

# Columns CatBoost will treat as categorical
CATEGORICAL_COLUMNS = [
    'season', 'mnth', 'hr', 'weekday', 'weathersit', 'holiday', 'workingday', 'yr'
]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def preprocess_for_catboost(df, drop_atemp=True, scale_numeric=True, add_cyclic=True):

    df = df.copy()

    #Remove datetime-like columns if present
    for col in ["dteday", "datetime", "date"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop atemp if requested
    if drop_atemp and 'atemp' in df.columns:
        df = df.drop(columns=['atemp'])

    # Drop unnecessary identifier column
    if 'instant' in df.columns:
        df = df.drop(columns=['instant'])

    # Create weekend feature if weekday exists
    if 'weekday' in df.columns:
        df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)

    # Cyclic encoding (optional)
    if add_cyclic:
        if 'hr' in df.columns:
            df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
            df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)

        if 'mnth' in df.columns:
            df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
            df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)

    # Scale numeric columns
    scaler = None
    numeric_cols = ['temp', 'hum', 'windspeed']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if scale_numeric and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


def prepare_xy(df, target='cnt', drop_cols_extra=['casual', 'registered']):
    df = df.copy()
    drop_cols = [target] + [c for c in drop_cols_extra if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]
    return X, y


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
