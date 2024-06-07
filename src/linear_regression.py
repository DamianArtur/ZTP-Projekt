import os
from typing import Tuple

import joblib

import schedule
import time
import pytz
import datetime
import warnings

from dask.dataframe import DataFrame
from dask.array import Array
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_absolute_error, mean_squared_error

import numpy as np

from read_file import read_and_clear_file

warnings.filterwarnings("ignore", category=FutureWarning)

def data_preparation(data: DataFrame) -> DataFrame:
    scaler = StandardScaler()
    scaler.fit(data)

    data = scaler.transform(data)
    return data


def convert_dataframe_to_arrays(data: DataFrame) -> Array:
    data = data.to_dask_array(lengths=True)
    return data


def split_data(data: Array) -> Tuple[Array, Array, Array, Array]:
    X = data[:, 0:-1]
    Y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=15, shuffle=True)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, X_test, y_train, y_test, model_file) -> None:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, model_file)

    y_pred = lr.predict(X_test).compute()

    time_utc = datetime.datetime.utcnow()
    current_time = time_utc.astimezone(pytz.timezone('Europe/Warsaw'))
    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    with open("./model/output_log.txt", "a+") as f:
        f.write(f"Date and time of execution: {current_time}\n")
        f.write(f'Accuracy (R^2): {round(lr.score(X_test, y_test), 2) * 100}%\n')
        f.write(f'MAE : {round(mean_absolute_error(y_test, y_pred), 2)}\n')
        f.write(f'RMSE : {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}\n\n\n')


def run_model(file, model_file) -> None:
    housing_data = read_and_clear_file(file)
    scaled_data = data_preparation(housing_data)

    data_array = convert_dataframe_to_arrays(scaled_data)

    X_training, X_testing, y_training, y_testing = split_data(data_array)

    linear_regression(X_training, X_testing, y_training, y_testing, model_file)


if __name__ == '__main__':
    file = os.path.abspath('data/apartments_pl_2024_01.csv')
    model_file = os.path.abspath('model/linear_regression_model.joblib')

    schedule.every().hour.do(run_model, file=file, model_file=model_file)

    run_model(file, model_file)

    while True:
        schedule.run_pending()
        time.sleep(1)
