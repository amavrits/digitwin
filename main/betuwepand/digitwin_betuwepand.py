import os
import pandas as pd
from digitwin.digitwin import DigiTwinBase
from geolib.models.dstability import DStabilityModel
from pathlib import Path
import dateutil.parser
from typing import List, Any
from create_scenarios import *
import subprocess


class DigiTwinBetuwepand(DigiTwinBase):

    def set_scenarios(self, data: dict, **kwargs) -> dict:

        scenarios = {}

        scenario_types = kwargs["scenario_types"]
        if not isinstance(scenario_types, list):
            scenario_types = list(scenario_types)

        scenario_types = [scenario_type.lower() for scenario_type in scenario_types]

        scenario_found = False
        if "daily" in scenario_types:
            time_max_measured = data["weather"][sensor_ids[-1]].idxmax()
            scenario = create_scenario_raai(
                scenario_name="Measurements "+time_max_measured.strftime('%Y-%m-%d %X'),
                time=time_max_measured,
                sensor_ids=data["sensor_ids"],
                input_df=data["weather"],
                sensor_df=data["sensors"],
                peil_df=data["piezometer"],
                canal_level=data["canal_level"],
                polder_level=data["polder_level"]
            )
            scenarios.update({scenario["name"]: scenario})
            scenario_found = True
        if "simulated" in scenario_types:
            time_max_hindcasted = data["weather"][sensor_ids[-1]]['Simulation'].idxmax()
            scenario = create_scenario_raai(
                scenario_name="Hindcasted from {: .0f}".format(time_max_hindcasted),
                time=time_max_hindcasted,
                sensor_ids=data["sensor_ids"],
                input_df=data["weather"],
                sensor_df=data["sensors"],
                peil_df=data["piezometer"],
                canal_level=data["canal_level"],
                polder_level=data["polder_level"]
            )
            scenarios.update({scenario["name"]: scenario})
            scenario_found = True
        if "return period" in scenario_types:
            if "return_period" in list(kwargs.keys()):
                return_period = kwargs["return_period"]
            else:
                raise Exception("Return periods not provided")
            scenario = create_scenario_return(
                scenario_name="{:.0f} year return period (extrapolated)".format(return_period),
                time=return_period,
                sensor_ids=data["sensor_ids"],
                input_df=data["weather"],
                return_df=data["returns"],
                sensor_df=data["sensors"],
                peil_df=data["piezometer"],
                canal_level=data["canal_level"],
                polder_level=data["polder_level"]
            )
            scenarios.update({scenario["name"]: scenario})
            scenario_found = True
        if "extreme value" in scenario_types:
            if "eva_times" in list(kwargs.keys()):
                eva_times = kwargs["eva_times"]
            else:
                raise Exception("EVA times not provided")
            for eva_time in eva_times:
                scenario = create_scenario_idf(
                    scenario_name="{:.0f year return period (idf)".format(eva_time),
                    time=eva_time,
                    sensor_ids=data["sensor_ids"],
                    input_df=data["weather"],
                    idf_df=data["extremes"],
                    sensor_df=data["sensors"],
                    peil_df=data["piezometer"],
                    canal_level=data["canal_level"],
                    polder_level=data["polder_level"]
                )
                scenarios.update({scenario["name"]: scenario})
                scenario_found = True
        if not scenario_found:
            raise Exception("Scenario not implemented.")

        return scenarios

    def inference(self):
        pass

    def predict(self):
        pass

    def performance(self, geo_model: Any, scenarios: dict) -> dict:

        for i_scenario, (scenario_name, scenario) in enumerate(scenarios.items()):

            PL = scenario['PL']
            HL = scenario['HL']
            geo_model = adapt_waternet(geo_model, PL, HL)

            # Calculate FoS and collect slip plane details.
            StabilityConsole = r"C:\Program Files (x86)\Deltares\D-GEO Suite\D-Stability 2024.01\bin\D-Stability Console.exe"
            dm.serialize(Path(r'.\work_folder\test.stix'))
            FileName = r".\work_folder\test.stix"
            cmd = ('"' + StabilityConsole + '" "' + FileName + '"')
            subprocess.call(cmd, shell=True)
            dm_copy = DStabilityModel()
            dm_copy.parse(Path('./work_folder/test.stix'))
            FoS = dm_copy.get_result().FactorOfSafety
            slip_plane = draw_lift_van_slip_plane(x_center_left=dm_copy.get_result().get_slipcircle_output().x_left,
                                                  z_center_left=dm_copy.get_result().get_slipcircle_output().z_left,
                                                  x_center_right=dm_copy.get_result().get_slipcircle_output().x_right,
                                                  z_center_right=dm_copy.get_result().get_slipcircle_output().z_right,
                                                  tangent_line=dm_copy.get_result().get_slipcircle_output().z_tangent,
                                                  surface_line={'s': surface_line[:, 0], 'z': surface_line[:, 1]})

            #TODO: PTK calculation

            scenario['FoS'] = FoS
            scenario['Slip plane'] = slip_plane
            scenarios[scenario_name] = scenario

        return scenarios

    def visualize(self):
        pass

    def export(self):
        pass
    
    def prepare_data(self, data: dict) -> dict:
        raai_df = self.sensor_data(data["sensors"])
        weather_df = self.weather_data(data["sensor_folder"], data["weather"])
        idf_head_res = self.extremes_data(data["extremes"])
        peil_df = self.piezometer_data(data["piezometer"])
        return_t_df_eva = self.return_data(data["returns"])
        data = {
            "sensors": raai_df,
            "weather": weather_df,
            "extremes": idf_head_res,
            "returns": return_t_df_eva,
            "piezometer": peil_df,
        }
        return data

    def sensor_data(self, sensor_path: str) -> pd.DataFrame:
        raai_df = pd.read_excel(sensor_path, sheet_name="Sensoren")
        return raai_df

    def combine_data(self, sensor_files: List[str], weather_file: pd.DataFrame) -> pd.DataFrame:
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

    def weather_data(self, sensor_folder:str, weather_path: str) -> pd.DataFrame:
        sensor_files = [sensor_folder + '\\' + sensor_file for sensor_file in os.listdir(sensor_folder)]
        input_data = self.combine_data(sensor_files, weather_path)
        return input_data

    def extremes_data(self, extremes_path: str) -> pd.DataFrame:
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

    def piezometer_data(self, piezometer_path: str) -> pd.DataFrame:
        peil_df = pd.read_csv(output_folder.joinpath(piezometer_path), index_col=0, header=[0, 1], parse_dates=True)
        peil_df = peil_df[peil_df.index >= dateutil.parser.parse("2021-05-01")]
        return peil_df

    def return_data(self, returns_path: str) -> pd.DataFrame:
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



if __name__ =="__main__":

    paths = {
        "sensors": r"./data/overview_raai_noord.xlsx",
        "sensor_folder": r'data/raw/piezometer data',
        "weather": r'data/processed/weather_knmi.pickle',
        "extremes": r"./work_folder/idf_extremes.pkl",
        "returns": r'data/processed/extremes.csv',
        "piezometer": r'test_peil_pred.csv',
    }

    output_folder = Path(r"./work_folder")

    return_ts = [2, 10, 20, 100, 200, 1000]
    sensor_ids = [
        'Raai_6B_Noord_Binnenkruin_(RFT_R52_0042_05m_BiKr)-20220901-20230417',
        'Raai_6B_Noord_Binnenteen_(RFT_R52_0042_05m_BiT)-20220901-20230417',
        'Raai_6B_Noord_Buitenkruin_(RFT_R52_0042_05m_BuKr)-20220901-20230417',
        'Raai_6B_Noord_Insteek_berm_(RFT_R52_0042_05m_InstBrm)-20220901-20230417',
        'PB-01 tm 2023-06'
    ]
    canal_level = np.array([[0, 5.6], [23.39, 5.6]])
    polder_level = np.array([[65.086, 1.7], [80, 1.7]])

    dt = DigiTwinBetuwepand()
    data = dt.prepare_data(paths)
    data.update({
        "sensor_ids": sensor_ids,
        "canal_level": canal_level,
        "polder_level": polder_level
    })

    return_period = 1_000
    eva_times = [10, 100, 1000]
    scenarios = dt.set_scenarios(
        data,
        scenario_types=["daily", "return period"],
        return_period=1_000,
        eva_times=eva_times
    )

    results = dt.performance(scenarios)

    # dm = DStabilityModel()
    # dm.parse(Path(r"./data/processed/template.stix"))
    # surface_line = get_surface_line(dm)

