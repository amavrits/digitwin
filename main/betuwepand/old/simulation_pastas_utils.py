import plotly.graph_objects as go
import plotly.io as io
import pastas as ps


def create_similation_model(input_data, pd_name, river_level_col, function, neerslag):
    obs_with_data = input_data[pd_name]
    verdamping = input_data['verdamping']
    river_level = input_data[river_level_col]
    if function is not ps.TarsoModel:
        ml = ps.Model(obs_with_data, name='water', freq='H')
        sm = ps.RechargeModel(neerslag,
                              verdamping,
                              rfunc=function()
                              )
        ml.add_stressmodel(sm)

    else:
        ml = ps.Model(obs_with_data, constant=False, name='water', freq='H')
        sm = ps.TarsoModel(neerslag,
                           verdamping,
                           obs_with_data)

        ml.add_stressmodel(sm)
    ml.add_stressmodel(ps.StressModel(river_level,
                                          rfunc=ps.One(),
                                          name='river_level',
                                          settings="waterlevel"))
    return ml


def simulate_model(input_data, pd_name, river_level_col, model, rainfall, ps_data, peil_df):
    """
    Helper function to create and simulate the model with specific rainfall data.
    """
    # Create the model
    ml = create_similation_model(input_data, pd_name, river_level_col, model, rainfall)

    # Set the model parameters
    for ii in range(len(ps_data.best_model_series[pd_name].parameters)):
        ml.set_parameter(ps_data.best_model_series[pd_name].parameters.iloc[ii].name,
                         initial=ps_data.best_model_series[pd_name].parameters.iloc[ii].optimal,
                         vary=False)

    # Simulate the model
    result = ml.simulate(tmin=peil_df.index[0], tmax=peil_df.index[-1])

    return result


