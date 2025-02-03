import numpy as np


def create_scenario_raai(scenario_name, time, sensor_ids, input_df, sensor_df, peil_df, canal_level, polder_level):
    PL_max_measured = []
    sensor_ids = ['KGM_065447_Maurikse Wetering_Hpol', 'KGM_065447_Maurikse Wetering_Hriv']  # FIXME
    for sensor_id in sensor_ids[:-1]:
        if sensor_id in input_df.columns:
            PL_max_measured.append([float(sensor_df.loc[sensor_df.Naam == sensor_id, 'local_x'].values),
                                    np.minimum(
                                        sensor_df[sensor_df.Naam == sensor_id]['Hoogte tov NAP'].values[0],
                                        input_df[sensor_id].loc[time]
                                    )])

    PL_max_measured = np.zeros((2, 2))  # FIXME
    PL_max_measured = np.concatenate((canal_level, PL_max_measured, polder_level))
    
    # FIXME
    HL_max_measured = PL_max_measured
    # temp = float(input_df[sensor_ids[-1]].loc[time])
    #
    # if np.isnan(temp):
    #     temp = peil_df[sensor_ids[4]]['Simulation'].loc[time]
    #
    # HL_max_measured = np.array([[0, temp]])
    
    return {
        'name': scenario_name,
        'date': time,
        'PL': PL_max_measured,
        'HL': HL_max_measured
    }


def create_scenario_return(scenario_name, time, sensor_ids, input_df, sensor_df, return_df, peil_df, canal_level, polder_level):

    # keep only the sensors with a simulation value
    # PL_max_measured = []
    # for sensor_id in sensor_ids[:-1]:
    #     if sensor_id in [i[0] for i in return_df.index]:
    #         if 'Simulation' in [i[1] for i in return_df.index if sensor_id == i[0]]:
    #             PL_max_measured.append([float(sensor_df[sensor_df.Naam == sensor_id]['local_x']),
    #                                     np.minimum(
    #                                         sensor_df[sensor_df.Naam == sensor_id]['Hoogte tov NAP'].values[0],
    #                                         return_df[sensor_id]['Simulation'][time]
    #                                     )])


    # PL_max_measured = np.concatenate((canal_level, PL_max_measured, polder_level))
    PL_max_measured = np.concatenate((canal_level, canal_level, polder_level))  #FIXME

    # HL_max_measured = np.array([[0, peil_df[sensor_ids[-1]]['Simulation'][time]]])
    HL_max_measured = np.array([[0, 1]]) #FIXME

    return {
        'name': scenario_name,
        'date': time,
        'PL': PL_max_measured,
        'HL': HL_max_measured
    }

def create_scenario_idf(scenario_name, time, sensor_ids, input_df, sensor_df, idf_df, peil_df, canal_level, polder_level):

    PL_max_measured = np.array([
        [float(sensor_df[sensor_df.Naam == sensor_id]['local_x']),
         np.minimum(
             sensor_df[sensor_df.Naam == sensor_id]['Hoogte tov NAP'].values[0],
             idf_df[idf_df.Scenario == sensor_id][time].values[0]
         )]
        for sensor_id in sensor_ids[:-1]
    ])

    PL_max_measured = np.concatenate((canal_level, PL_max_measured, polder_level))

    HL_return_period_idf = np.array([[0, idf_df[idf_df.Scenario == sensor_ids[-1]][time].values[0]]])

    return {
        'name': scenario_name,
        'date': time,
        'PL': PL_max_measured,
        'HL': HL_return_period_idf
    }

def create_preliminary_scenarios(time_series):
    # three scenarios are defined: the maximum measured water level, the minimum measured water level and the return period
    # water level
    # get index of maximum water level when column OW000449 is maximum
    max_index = time_series['OW000449'].idxmax()
    # get full series of maximum water level
    max_series = time_series.loc[max_index]
    x_A = [max_series[counter] for counter in range(len(max_series)) if "A_" in max_series.index[counter] and "x" in max_series.index[counter]]
    y_A = [max_series[counter] for counter in range(len(max_series)) if "A_" in max_series.index[counter] and "y" in max_series.index[counter]]
    x_B = [max_series[counter] for counter in range(len(max_series)) if "B_" in max_series.index[counter] and "x" in max_series.index[counter]]
    y_B = [max_series[counter] for counter in range(len(max_series)) if "B_" in max_series.index[counter] and "y" in max_series.index[counter]]
    max_scenario = {
        'name': 'max',
        'date': max_index,
        'HL 1': list(zip(x_A, y_A)),
        'HL 2': list(zip(x_B, y_B))

    }
    # get index of minimum water level when column OW000449 is minimum
    min_index = time_series['OW000449'].idxmin()
    # get full series of minimum water level
    min_series = time_series.loc[min_index]
    x_A = [min_series[counter] for counter in range(len(min_series)) if "A_" in min_series.index[counter] and "x" in min_series.index[counter]]
    y_A = [min_series[counter] for counter in range(len(min_series)) if "A_" in min_series.index[counter] and "y" in min_series.index[counter]]
    x_B = [min_series[counter] for counter in range(len(min_series)) if "B_" in min_series.index[counter] and "x" in min_series.index[counter]]
    y_B = [min_series[counter] for counter in range(len(min_series)) if "B_" in min_series.index[counter] and "y" in max_series.index[counter]]
    min_scenario = {
        'name': 'min',
        'date': min_index,
        'HL 1': list(zip(x_A, y_A)),
        'HL 2': list(zip(x_B, y_B))
    }
    # get the index closest to the median of the return period water level
    return_index = time_series['OW000449'].sub(time_series['OW000449'].median()).abs().idxmin()
    # get full series of return period water level
    return_series = time_series.loc[return_index]
    x_A = [return_series[counter] for counter in range(len(return_series)) if "A_" in return_series.index[counter] and "x" in return_series.index[counter]]
    y_A = [return_series[counter] for counter in range(len(return_series)) if "A_" in return_series.index[counter] and "y" in return_series.index[counter]]
    x_B = [return_series[counter] for counter in range(len(return_series)) if "B_" in return_series.index[counter] and "x" in return_series.index[counter]]
    y_B = [return_series[counter] for counter in range(len(return_series)) if "B_" in return_series.index[counter] and "y" in return_series.index[counter]]
    return_scenario = {
        'name': 'return',
        'date': return_index,
        'HL 1': list(zip(x_A, y_A)),
        'HL 2': list(zip(x_B, y_B))
    }
    scenarios_dict = {
        'max': max_scenario,
        'min': min_scenario,
        'return': return_scenario
    }

    return scenarios_dict