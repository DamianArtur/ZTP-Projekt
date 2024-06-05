from typing import Generator, Tuple, Any

from streamz import Stream
import joblib

from dask_ml.linear_model import LinearRegression
from dask.dataframe import DataFrame

import numpy as np
import time

from read_file import read_and_clear_file

model_path = '../model/linear_regression_model.joblib'
file = '../data/apartments_pl_2024_01.csv'
output_file = '../data/predictions.csv'

def load_model(model_path) -> LinearRegression:
    return joblib.load(model_path)

def generate_stream_data(file) -> Generator[Tuple[DataFrame], None, None]:
    data_frame = read_and_clear_file(file)
    data_array = data_frame.to_dask_array(lengths=True)
    data_array = data_array[:, 0:-1]

    indices = np.random.choice(len(data_array), size=10, replace=False)
    selected_rows = data_array[indices]

    for row in selected_rows:
        yield row.reshape(1, -1)

def process_stream_data(data) -> Tuple[np.ndarray, Any]:
    model = load_model(model_path)
    prediction = model.predict(data).compute()
    return data, prediction

def save_predictions(predictions) -> None:
    with open(output_file, 'a') as f:
        for data, prediction in predictions:
            data_values = data.compute().flatten().tolist()
            f.write(f"{data_values},{prediction}\n")


if __name__ == '__main__':
    stream = Stream()
    predictions = []

    stream.map(process_stream_data).sink(lambda x: predictions.append(x))

    data_generator = generate_stream_data(file)

    for new_data in data_generator:
        stream.emit(new_data)
        time.sleep(0.1)

    save_predictions(predictions)
