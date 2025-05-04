# utils.py

import pandas as pd
import numpy as np
from datetime import timedelta
import holidays

IN_HOLIDAYS = holidays.India()

def create_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['holidays'] = df['date'].isin(IN_HOLIDAYS).astype(int)
    df['m1'] = np.sin(2 * np.pi * df['month'] / 12)
    df['m2'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def get_next_dates(last_date, n):
    return [last_date + timedelta(days=i) for i in range(1, n + 1)]

def prepare_recursive_forecast_input(df, model, store_id, item_id):
    df = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(), []

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = create_features(df)

    # Get last 3 known sales values
    last_sales = df['sales'].tail(3).tolist()
    if len(last_sales) < 3:
        return pd.DataFrame(), pd.Series(), []

    lag_1, lag_2, lag_3 = last_sales[-1], last_sales[-2], last_sales[-3]
    last_date = df['date'].max()

    forecasts = []
    forecast_dates = []

    for i in range(7):
        next_date = last_date + timedelta(days=1)
        forecast_dates.append(next_date)

        feature_df = pd.DataFrame([{
            'date': next_date,
            'store': store_id,
            'item': item_id,
            'sales_lag_1': lag_1,
            'sales_lag_2': lag_2,
            'sales_lag_3': lag_3
        }])

        feature_df = create_features(feature_df)
        feature_cols = ['store', 'item', 'year', 'month', 'day', 'weekday',
                        'holidays', 'm1', 'm2', 'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'weekend']
        X = feature_df[feature_cols]

        pred = model.predict(X)[0]
        forecasts.append(pred)

        # Update lags
        lag_3, lag_2, lag_1 = lag_2, lag_1, pred
        last_date = next_date

    return forecast_dates, forecasts
