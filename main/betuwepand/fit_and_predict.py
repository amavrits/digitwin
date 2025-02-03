import pandas as pd
from pathlib import Path
import numpy as np
import pastas as ps
import matplotlib.pyplot as plt

ps.set_log_level("ERROR")


def read_pastas_models(files):
    """
    Read Pastas models from files and store them in a Pandas Series.

    Parameters:
    - files (list or iterable): List of file paths for Pastas model files.

    Returns:
    - pastas_series (pd.Series): Pandas Series containing Pastas models, indexed by file names.

    """
    # Create an empty dictionary to store Pastas models
    pastas_dict = {}

    # Iterate through each file path in the input list
    for f in files:
        # Convert the file path to a Path object
        f = Path(f)

        # Load the Pastas model from the file and store it in the dictionary with the file name as the key
        pastas_dict[f.stem] = ps.io.load(f)

    # Create a Pandas Series from the dictionary
    pastas_series = pd.Series(pastas_dict)

    # Return the Pandas Series containing Pastas models
    return pastas_series


class PastasUtils:
    """
    A utility class for working with Pastas models and data.

    Attributes:
        basepaths (basepaths): An instance of the basepaths class containing directory paths.
        rfuncs (list of rfunc): A list of Pastas stress response functions.
        pastas_folder (Path): The path to the Pastas directory for saving models and results.
        all_fitted_models_path (Path): The path to the file storing all fitted Pastas models.
        best_fitted_model_path (Path): The path to the file storing the best Pastas models.
        use_waterlevels (bool): Flag indicating whether water levels should be used.
        prob_alpha (float or None): The significance level for prediction intervals (e.g., 0.05).
        on_metric (str): The evaluation metric for selecting the best models.
        again (bool): Flag indicating whether to recompute models.
        all_result_df (pd.DataFrame): DataFrame containing model results.
        best_model_series (pd.Series): Series containing the best models for each observation.

    Methods:
        fit_pastas_to_peilen(input_data)
        find_best_pastas_models(on_metric='rmse')
        fit_all_pastas_models(input_data, use_waterlevels=False)
        get_periods_with_inputdata(input_data, input_cols)
        get_predictions_for_all_locations(tmin, tmax)
        predict(model, tmin, tmax, prob_alpha=None)
        get_highest_date_from_timeseries(peil_timeseries)
    """

    def __init__(self, input_data, neerslag_col, verdamping_col, peilbuis_cols, prob_alpha=None, on_metric='rmse', rfuncs=None, river_level = None):
        """
        Initialize the PastasUtils object and fit Pastas models to the provided data.

        Args:
            input_data (pd.DataFrame): input_data for pastas.
            prob_alpha (float or None): The significance level for prediction intervals.
            on_metric (str): The evaluation metric for selecting the best models.
            rfuncs (list of rfunc): List of Pastas stress response functions.
            river_level (str): The name of the river level column in the input data if it exists.
        """
        self.rfuncs = rfuncs if rfuncs else [ps.rfunc.Gamma, ps.rfunc.Exponential, ps.rfunc.Polder,
                                             ps.rfunc.FourParam, ps.rfunc.DoubleExponential, ps.rfunc.Edelman,
                                             ps.TarsoModel]        
        
        self.prob_alpha = prob_alpha
        self.on_metric = on_metric
        if self.on_metric not in ps.stats.metrics.__all__:
            raise Exception(f"{self.on_metric} is not available in pastas, choose one of {ps.stats.metrics.__all__}")
        
        # Fit Pastas models to peilen
        self.fit_pastas_to_peilen(input_data, neerslag_col, verdamping_col, peilbuis_cols, river_level)

    def fit_pastas_to_peilen(self, input_data, neerslag_col, verdamping_col, peilbuis_cols, river_level):
        """
        Fit Pastas models to the observed data based on the provided settings and data.
        
        This method determines whether to reuse existing results or compute new models based on the current settings.
        
        Args:
            input_data (pd.DataFrame): The input data for model fitting.
            basepaths (BasePaths): An object containing paths to store/retrieve results.
            
        Returns:
            None
        """
        # Get periods with input data based on water levels usage
        input_cols = [neerslag_col, verdamping_col]
        self.get_periods_with_inputdata(input_data, input_cols)
    
        self.fit_all_pastas_models(input_data, neerslag_col, verdamping_col, peilbuis_cols, river_level)
        self.find_best_pastas_models(on_metric=self.on_metric)
   
    def fit_all_pastas_models(self, input_data, neerslag_col, verdamping_col, peilbuis_cols, river_level):
        """
        Fit Pastas models to all observation series based on the provided input data and settings.
        
        This method fits Pastas models to all observation series and stores the results in self.all_result_df.
        
        Args:
            input_data (pd.DataFrame): The input data for model fitting.
        
        Returns:
            None
        """
        peilbuizen = self.selected_input_data[peilbuis_cols]
        neerslag = self.selected_input_data[neerslag_col]
        verdamping = self.selected_input_data[verdamping_col]
        if river_level is not None:
            river_level = self.selected_input_data[river_level]
        
        self.all_result_df = pd.DataFrame()
        for peilbuis_naam in peilbuizen.columns:
            # print the progress
            print(f'Fitting Pastas models for {peilbuis_naam} out of {peilbuizen.columns.size} peilbuizen.')
            obs = peilbuizen[peilbuis_naam]
            data_idx = obs[obs.notnull()].index
            obs_with_data = obs[data_idx[0]:data_idx[-1]]
                     
            results = []
            for rfunc in self.rfuncs:
                if rfunc is not ps.TarsoModel:
                    ml = ps.Model(obs_with_data, name='water',freq='H')
                    sm = ps.RechargeModel(neerslag, 
                                          verdamping,  
                                          rfunc=rfunc()
                                          )
                    ml.add_stressmodel(sm)
                    
                else:
                    ml = ps.Model(obs_with_data, constant=False, name='water', freq='H')
                    sm = ps.TarsoModel(neerslag, 
                                       verdamping, 
                                       obs_with_data)
          
                    ml.add_stressmodel(sm)
                if river_level is not None:
                    ml.add_stressmodel(ps.StressModel(river_level,
                                                      rfunc=ps.One(),
                                                      name='river_level',
                                                      settings="waterlevel"))
                
                # ml.solve(noise=False, initial = True, report = False)
                ml.del_noisemodel()
                try:
                    ml.solve( initial=True, report=False)
                except ValueError as e:
                    print(f"Could not solve model for {peilbuis_naam} with rfunc {rfunc} skipping this rfunc.")
                    print(e)
                results.append(np.r_[[rfunc], [ml], [getattr(ml.stats, s)() for s in ps.stats.metrics.__all__]])
       
            results = np.array(results)
            result_df = pd.DataFrame(results[:, 1:], index=results[:, 0], columns=['model']+ps.stats.metrics.__all__)
            self.all_result_df = pd.concat([self.all_result_df, result_df])
        
        # Add 'oseries' column to identify the observation series
        self.all_result_df.index = pd.MultiIndex.from_arrays([[m.oseries.name for m in self.all_result_df.model], self.all_result_df.index], names = ['peilbuis','rfunc'])
        
    def find_best_pastas_models(self, on_metric='rmse'):
        """
        Find the best Pastas models for each observation based on a specified evaluation metric.
        
        This method analyzes the model results and selects the best model for each observation based on the given metric.
        
        Args:
            on_metric (str): The evaluation metric to determine the best models (default is 'rmse').
        
        Returns:
            None
        """
        best_models = {}
        best_model_functions = {}
        
        # Group model results by observation name
        for obsname, obsdata in self.all_result_df.groupby(level='peilbuis'):
            # Find the model with the lowest value for the specified metric
            best_performing = obsdata[obsdata[on_metric] == obsdata[on_metric].min()].iloc[0].model
            model_function = obsdata[obsdata[on_metric] == obsdata[on_metric].min()].index.values[0][1]
            # add to the series as a new column
            best_model_functions[obsname] = model_function
            best_models[obsname] = best_performing
        
        # Create a Pandas Series to store the best models
        self.best_model_series = pd.Series(best_models)
        self.best_model_functions = pd.Series(best_model_functions)
            
    def get_periods_with_inputdata(self, input_data, input_cols):
        """
        Select periods with complete input data based on the specified settings.
    
        This method filters the input data and selects periods with complete data.
    
        Args:
            input_data (pd.DataFrame): The input data for model fitting.
            input_cols (list): list of columns names where the forcing data (precipitation, evaporation) can be found
    
        Returns:
            None
        """
    
        # Filter and select periods with complete input data
        input_idxs = input_data.index[input_data[input_cols].notnull().all(1)]
        self.selected_input_data = input_data.loc[input_idxs]
        
    def get_predictions_for_all_locations(self, tmin, tmax):
        """
        Get predictions for all locations using the best Pastas models.
    
        This method retrieves predictions for all locations based on the best Pastas models.
    
        Args:
            tmin (pd.Timestamp): The start time for predictions.
            tmax (pd.Timestamp): The end time for predictions.
    
        Returns:
            pd.DataFrame: A DataFrame containing predictions for all locations.
        """
        peil_df = pd.DataFrame()
    
        # Iterate over the best models for each location
        for name, model in self.best_model_series.items():
            peil = self.predict(model, tmin, tmax, prob_alpha=self.prob_alpha)
            # Add location name as a column level and concatenate predictions
            peil.columns = pd.MultiIndex.from_product([[name], peil.columns], names = ['peilbuis', 'simname'])
            peil_df = pd.concat([peil_df, peil], axis=1)
    
        return peil_df
        
    def predict(self, model, tmin, tmax, prob_alpha=None):
        """
        Generate predictions using the specified Pastas model.
    
        This method generates predictions based on the provided Pastas model within the specified time range.
    
        Args:
            model (ps.Model): The Pastas model for making predictions.
            tmin (pd.Timestamp): The start time for predictions.
            tmax (pd.Timestamp): The end time for predictions.
            prob_alpha (float or None): The significance level for prediction intervals (e.g., 0.05).
    
        Returns:
            pd.DataFrame: Predictions or prediction intervals based on the model and specified time range.
        """

        peil = model.simulate(tmin=tmin, tmax=tmax).to_frame()
        
        if prob_alpha is not None:
            # Generate prediction intervals
            prob_peil = model.fit.prediction_interval(alpha=prob_alpha, n=1000, tmin=tmin, tmax=tmax)
            peil = pd.concat([peil, prob_peil], axis=1)
        return peil
    
    def save_best_pastas_fit(self, to_folder):
        """
        Save the best Pastas fits to files.
    
        Parameters:
        - to_folder (str or pathlib.Path): Path to the folder where Pastas model files will be saved.
    
        Returns:
        None (Saves Pastas models to files).
    
        """
        # Iterate through each name and corresponding best model in the series
        for name, model in self.best_model_series.items():
            # Save the Pastas model to a file in the specified folder
            model.to_file(Path(to_folder) / f'{name}.pas')

    
    def get_highest_date_from_timeseries(self, peil_timeseries):
        """
        Find the time series with the highest normalized average value from a given DataFrame.
    
        This method identifies the time series with the highest normalized average value from the provided DataFrame.
    
        Args:
            peil_timeseries (pd.DataFrame): DataFrame containing time series data.
    
        Returns:
            pd.Series: The time series with the highest normalized average value.
        """
        # Select time series with no missing values
        sel_df_wl = peil_timeseries[peil_timeseries.notnull().all(1)]
    
        # Normalize the selected time series
        min_value = sel_df_wl.min()
        max_value = sel_df_wl.max()
        normalized_cross_peilen = (sel_df_wl - min_value) / (max_value - min_value)
    
        # Find the index corresponding to the highest normalized average
        normalized_highest = normalized_cross_peilen.mean(1).idxmax()
    
        # Get the time series with the highest normalized average
        highest_peilen = sel_df_wl.loc[normalized_highest]
    
        return highest_peilen


#%%
if __name__ == '__main__':
    input_data = pd.read_csv(r'C:\Users\meer_an\repositories\purmerfloodingprediction\scenarios\purmer\data\processed\alldata.csv', index_col = 0, parse_dates = True)
    #input_data = pd.read_csv(r'../../../../data/processed/alldata.csv', index_col=0, parse_dates=True)
    pastas_model_folder = r'C:\Users\meer_an\repositories\purmerfloodingprediction\scenarios\purmer\work_folder'#'pastas_saves'
    neerslag_col = 'neerslag'
    verdamping_col = 'verdamping'
    peilbuis_cols = ['PB007718', 'HB-6B-1_PB1', 'HB-6B-2_PB1', 'HB-6B-3_PB1', 'HB-6B-4_PB1']
    peilbuis_cols = ['HB-6B-3_PB1']
    
    prob_alpha = 0.05
    
    ps_data = PastasUtils(input_data, 
                     neerslag_col, 
                     verdamping_col, 
                     peilbuis_cols, 
                     prob_alpha=prob_alpha, 
                     on_metric='mae', 
                     rfuncs=None)
    
    ps_data.save_best_pastas_fit(pastas_model_folder)
    
    tmin = ps_data.selected_input_data.index.min()
    tmax = ps_data.selected_input_data.index.max()
    peil_df = ps_data.get_predictions_for_all_locations(tmin, tmax)
    peil_df.to_csv('test_peil_pred.csv')
    
    ####plotting
    fig, ax = plt.subplots(dpi = 600)
    i = 0
    colors = plt.get_cmap('tab10')
    for pname, pdata in peil_df.T.groupby(level='peilbuis'):
        pdata.index = pdata.index.droplevel('peilbuis')
        input_obs = input_data[pname]
        input_obs = input_obs[input_obs.notnull()]
        if input_obs.size > 200:
            input_obs = input_obs.iloc[np.arange(0, input_obs.size, input_obs.size/200).astype(int)]
        input_obs.plot(ax=ax, marker = 'x', linestyle = '', color = colors(i), alpha = 0.5, label = 'observations', legend = i==0)
        pdata.loc['Simulation'].plot(ax=ax, color = colors(i), label = 'pastas best fit', legend = i==0)
        ax.fill_between(pdata.columns, pdata.loc[prob_alpha/2], pdata.loc[1-prob_alpha/2], alpha = 0.1, color = colors(i), label = 'pastas 95% uncertainty')
        ax.set_xlim(input_obs.index.min(), input_obs.index.max())
        i+=1
    ax.set_ylabel('stijghoogte (m)')
    ax.legend(handles = ax.legend().legendHandles[:3], labels = [b.get_text() for b in ax.legend().texts[:3]])
    #label
        
        
        