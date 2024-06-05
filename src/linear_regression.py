from typing import Tuple

from dask.dataframe import DataFrame
from dask.array import Array
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from read_file import read_and_clear_file


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


def linear_regression(X_train, X_test, y_train, y_test) -> None:
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test).compute()
    print(f'Accuracy (R^2): {round(lr.score(X_test, y_test), 2) * 100}%')
    print(f'MAE : {round(mean_absolute_error(y_test, y_pred), 2)}')
    print(f'RMSE : {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}')


if __name__ == '__main__':
    file = '../data/apartments_pl_2024_01.csv'

    housing_data = read_and_clear_file(file)
    scaled_data = data_preparation(housing_data)

    data_array = convert_dataframe_to_arrays(scaled_data)

    X_training, X_testing, y_training, y_testing = split_data(data_array)

    linear_regression(X_training, X_testing, y_training, y_testing)
