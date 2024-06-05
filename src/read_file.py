from dask.dataframe import read_csv, DataFrame


def read_and_clear_file(file_name: str) -> DataFrame:
    # Read from a file
    data_frame = read_csv(file_name)

    # Delete unused columns

    cleared_dataframe = data_frame.drop(columns=[
        'id',
        'city',
        'type',
        # 'floorCount',
        # 'latitude',
        # 'longitude',
        # 'centreDistance',
        # 'poiCount',
        # 'schoolDistance',
        # 'clinicDistance',
        # 'postOfficeDistance',
        # 'kindergartenDistance',
        # 'restaurantDistance',
        # 'collegeDistance',
        # 'pharmacyDistance',
        'ownership',
        'buildingMaterial',
        'condition',
        # 'hasParkingSpace',
        # 'hasBalcony',
        # 'hasElevator',
        # 'hasSecurity',
        # 'hasStorageRoom'
    ])

    # Clear empty rows
    cleared_dataframe = cleared_dataframe.dropna()

    value_dataframe = cleared_dataframe.replace({'no': 0.0, 'yes': 1.0})
    value_dataframe = value_dataframe.astype(float)

    return value_dataframe


if __name__ == '__main__':
    file = '../data/apartments_pl_2024_01.csv'
    df = read_and_clear_file(file)
    print(df.head())
