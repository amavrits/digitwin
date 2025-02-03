import pandas as pd
from digitwin.digitwin import DigiTwinBase
import subprocess
from data_processing import *
from create_scenarios import *
from dike_stability_analysis import *
from typing import Any


class DigiTwinBetuwepand(DigiTwinBase):

    def itertimes(self, data: tuple, **kwargs) -> dict:

        if "return_period" in list(kwargs.keys()):
            return_period = kwargs["return_period"]
        else:
            raise Exception("Return periods not provided")

        if "eva_times" in list(kwargs.keys()):
            eva_times = kwargs["eva_times"]
        else:
            raise Exception("EVA times not provided")

        scenario_types = ["simulated", "return period", "extreme value"]
        # scenarios = self.set_scenarios(data, scenario_types, return_period=return_period, eva_times=eva_times)

        scenarios = {}
        daily_scenarios = {}
        calculation_days = list(data["weather"].index)
        calculation_days = calculation_days[:5]
        for calculation_day in calculation_days:
            scenario = self.set_scenarios(data, scenario_types=["daily"], calculation_day=calculation_day)
            daily_scenarios[calculation_day] = scenario

        scenarios.update(daily_scenarios)

        scenarios = self.performance(
            scenarios,
            stability_console=kwargs["stability_console"],
            file_path=kwargs["file_path"]
        )

        return scenarios


    def set_scenarios(self, data: dict, **kwargs) -> dict:

        scenarios = {}

        scenario_types = kwargs["scenario_types"]
        if not isinstance(scenario_types, list):
            scenario_types = list(scenario_types)

        scenario_types = [scenario_type.lower() for scenario_type in scenario_types]

        scenario_found = False
        if "daily" in scenario_types:
            if "calculation_day" in list(kwargs.keys()):
                calculation_day = kwargs["calculation_day"]
            else:
                calculation_day = data["weather"][sensor_ids[-1]].idxmax()
            scenario = create_scenario_raai(
                scenario_name="Measurements "+calculation_day.strftime('%Y-%m-%d %X'),
                time=calculation_day,
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

    def performance(self, scenario: dict, **kwargs) -> dict:

        stability_console = kwargs["stability_console"]
        file_path = kwargs["file_path"]

        geo_model = DStabilityModel()
        geo_model.parse(Path(file_path))
        surface_line = get_surface_line(geo_model)

        for i_scenario, (scenario_name, scenario) in enumerate(scenarios.items()):

            PL = scenario['PL']
            HL = scenario['HL']
            geo_model = adapt_waternet(geo_model, PL, HL)

            # Calculate FoS and collect slip plane details.
            geo_model.serialize(Path(file_path))
            cmd = ('"' + stability_console + '" "' + file_path + '"')
            subprocess.call(cmd, shell=True)

            geo_model_copy = DStabilityModel()
            geo_model_copy.parse(Path('./work_folder/test.stix'))
            fos = geo_model_copy.get_result().FactorOfSafety
            slip_plane = draw_lift_van_slip_plane(x_center_left=geo_model_copy.get_result().get_slipcircle_output().x_left,
                                                  z_center_left=geo_model_copy.get_result().get_slipcircle_output().z_left,
                                                  x_center_right=geo_model_copy.get_result().get_slipcircle_output().x_right,
                                                  z_center_right=geo_model_copy.get_result().get_slipcircle_output().z_right,
                                                  tangent_line=geo_model_copy.get_result().get_slipcircle_output().z_tangent,
                                                  surface_line={'s': surface_line[:, 0], 'z': surface_line[:, 1]})

            #TODO: PTK calculation

            scenario['FoS'] = fos
            scenario['Slip plane'] = slip_plane
            scenarios[scenario_name] = scenario

        return scenarios

    def visualize(self):
        pass

    def export(self):
        pass
    
    def prepare_data(self, data: dict) -> dict:
        raai_df = sensor_data(data["sensors"])
        weather_df = weather_data(data["sensor_folder"], data["weather"])
        idf_head_res = extremes_data(data["extremes"])
        peil_df = piezometer_data(data["piezometer"])
        return_t_df_eva = return_data(data["returns"])
        data = {
            "sensors": raai_df,
            "weather": weather_df,
            "extremes": idf_head_res,
            "returns": return_t_df_eva,
            "piezometer": peil_df,
        }
        return data


if __name__ =="__main__":

    paths = {
        "sensors": r"./data/overview_raai_noord.xlsx",
        "sensor_folder": r'data/raw/piezometer data',
        "weather": r'data/processed/weather_knmi.pickle',
        "extremes": r"./work_folder/idf_extremes.pkl",
        "returns": r'data/processed/extremes.csv',
        "piezometer": r'./work_folder/test_peil_pred.csv',
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
    stability_console = r"C:\Program Files (x86)\Deltares\D-GEO Suite\D-Stability 2024.01\bin\D-Stability Console.exe"
    file_path = r"work_folder/test.stix"

    results = dt.itertimes(
        data,
        eva_times=eva_times,
        return_period=return_period,
        stability_console=stability_console,
        file_path=file_path
    )

