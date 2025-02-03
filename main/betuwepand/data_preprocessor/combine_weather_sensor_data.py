import os
import pandas as pd
import pickle

def combine_data(sensor_files, weather_file):

    datetime_format = '%Y-%m-%d %H:%M'

    df_weather = pd.read_pickle(weather_file)
    df_weather = df_weather.rename(columns={'RH': 'neerslag', 'EV24': 'verdamping'})
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], errors='coerce', utc=True)
    df_weather.index = df_weather['datetime']
    df_weather = df_weather[['neerslag', 'verdamping']]
    df_weather = df_weather[~pd.isna(df_weather['neerslag'])]
    df_weather['neerslag'] /= 10  # In KNMI, rainfall data is in 0.1mm units
    df_weather = df_weather.tz_localize(None)

    dfs_sensor = []
    for i, sensor_file in enumerate(sensor_files):
        sensor_name = sensor_file.split('.')[-2].split('\\')[-1]
        df_sensor = pd.read_csv(sensor_file, delimiter=';')
        df_sensor['MessageTimestamp'] = pd.to_datetime(df_sensor['MessageTimestamp'], errors='coerce', utc=True)
        df_sensor['MessageTimestamp'] = df_sensor['MessageTimestamp'].dt.floor('h')
        df_sensor.index = df_sensor['MessageTimestamp']
        df_sensor = df_sensor['WaterLevel'].to_frame()
        df_sensor = df_sensor.rename(columns={'WaterLevel': sensor_name})
        df_sensor = df_sensor.sort_index()
        df_sensor = df_sensor.groupby(df_sensor.index).mean()
        dfs_sensor.append(df_sensor)
    df_sensor = pd.concat(dfs_sensor, axis=1)
    df_sensor = df_sensor.tz_localize(None)

    df = pd.concat((df_weather, df_sensor), axis=1)
    df = df.dropna(how='all')
    df.index = pd.to_datetime(df.index, format=datetime_format)

    return df


if __name__ == "__main__":

    sensor_folder = r'..\data\raw\piezometer data'
    sensor_files = [sensor_folder + '\\' + sensor_file for sensor_file in os.listdir(sensor_folder)]

    weather_file = r'..\data\processed\weather_knmi.pickle'

    df = combine_data(sensor_files, weather_file)
