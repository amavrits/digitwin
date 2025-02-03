import os
import pandas as pd
import dateutil.parser
from typing import List


def sensor_data(sensor_path: str) -> pd.DataFrame:
    raai_df = pd.read_excel(sensor_path, sheet_name="Sensoren")
    return raai_df


def combine_data(sensor_files: List[str], weather_file: str) -> pd.DataFrame:
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


def weather_data(sensor_folder: str, weather_path: str) -> pd.DataFrame:
    sensor_files = [sensor_folder + '\\' + sensor_file for sensor_file in os.listdir(sensor_folder)]
    input_data = combine_data(sensor_files, weather_path)
    return input_data


def extremes_data(extremes_path: str) -> pd.DataFrame:
    idf_head_res = pd.read_pickle(extremes_path)  # TODO: Adjust this for ARK
    idf_head_res = idf_head_res.iloc[:5]
    idf_head_res['Scenario'] = [  # TODO: Remove renaming of scenarios when the sensor data file becomes available
        'PB-01 tm 2023-06',
        'Raai_6B_Noord_Binnenkruin_(RFT_R52_0042_05m_BiKr)-20220901-20230417',
        'Raai_6B_Noord_Binnenteen_(RFT_R52_0042_05m_BiT)-20220901-20230417',
        'Raai_6B_Noord_Buitenkruin_(RFT_R52_0042_05m_BuKr)-20220901-20230417',
        'Raai_6B_Noord_Insteek_berm_(RFT_R52_0042_05m_InstBrm)-20220901-20230417'
    ]
    return idf_head_res


def piezometer_data(piezometer_path: str) -> pd.DataFrame:
    peil_df = pd.read_csv(piezometer_path, index_col=0, header=[0, 1], parse_dates=True)
    peil_df = peil_df[peil_df.index >= dateutil.parser.parse("2021-05-01")]
    return peil_df


def return_data(returns_path: str) -> pd.DataFrame:
    return_t_df_eva = pd.read_csv(returns_path)
    return_t_df_eva['peilbuis'].replace({
        'HB-6B-1_PB1': 'Raai_6B_Noord_Binnenkruin_(RFT_R52_0042_05m_BiKr)-20220901-20230417',
        'HB-6B-2_PB1': 'Raai_6B_Noord_Binnenteen_(RFT_R52_0042_05m_BiT)-20220901-20230417',
        'HB-6B-3_PB1': 'Raai_6B_Noord_Buitenkruin_(RFT_R52_0042_05m_BuKr)-20220901-20230417',
        'HB-6B-4_PB1': 'Raai_6B_Noord_Insteek_berm_(RFT_R52_0042_05m_InstBrm)-20220901-20230417',
        'PB007718': 'PB-01 tm 2023-06'
    }, inplace=True)
    return_t_df_eva.index = pd.MultiIndex.from_frame(return_t_df_eva[['peilbuis', 'simname', 'T']])
    return_t_df_eva = return_t_df_eva['return value']
    return return_t_df_eva


if __name__ == "__main__":

    pass
