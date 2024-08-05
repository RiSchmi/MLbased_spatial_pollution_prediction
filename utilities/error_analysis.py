from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class spatial_error_analysis():
    '''### Script for visual error analysis:
    `Class Input:`
    - df_models: model performance with attributes: ['y_true', 'y_pred', 'id']
    - names_models: list of model names, same length as df_models
    
    `Functions:`
    - Regression Metrics: MAE, RMSE, MAPE, R2
    - Plot. prediction over time per station
    - Plot. residual plot per model & station

    '''
    def __init__(self, df_models, names_models):

        if len(df_models) != len(names_models):
            print('n(models)! n(model_names)')  
        
        # define dict with models
        self.datasets_by_model = {}
        for n in range(len(df_models)):
            self.datasets_by_model[f'{names_models[n]}'] = df_models[n]
        
        self.ids = list(df_models[0].id.unique())
        self.n_models = len(names_models)
        self.n_ids = len(self.ids)

        # define datasets
        self.dict_datasets_by_id = self.get_dicts_by_id()
        self.dict_datasets_all = self.get_dict_all_models()

    def get_dict_all_models(self):
        'create dict size models * ids, dfs as values, filtered by id and model'
        dict_datasets_all = {}
        for model_name, df in self.datasets_by_model.items():
            for unique_id in self.ids:
                var_name = f"{model_name}_{unique_id}"
                globals()[var_name] = df[df['id'] == unique_id]
                dict_datasets_all[var_name] = globals()[var_name]
        return dict_datasets_all   

    def get_dicts_by_id(self):
        
        dict_datasets_by_id = {}
        for unique_id in self.ids:
            filtered_df_names = []
            for model_name, df in self.datasets_by_model.items():
                var_name = f"{model_name}_{unique_id}"
                globals()[var_name] = df[df['id'] == unique_id]
                filtered_df_names.append(globals()[var_name])
            dict_datasets_by_id[unique_id] = filtered_df_names
        return dict_datasets_by_id   

    def cal_acc(self, y_grt, y_prd):
        'calc performance metrics'
        mae = mean_absolute_error(y_grt, y_prd)
        mape = mean_absolute_percentage_error(y_grt, y_prd) 
        rmse = root_mean_squared_error(y_grt, y_prd)
        r2 = r2_score(y_grt, y_prd)
        mdape = np.median((np.abs(np.subtract(y_grt, y_prd)/ y_grt))) 

        return round(mae,3), round(rmse,3), round(mape, 3), round(mdape, 3), round(r2, 3), 
  
    def plot_predict_over_time(self, title, y_label, n_time_steps = 300, save = False, path = 'true_predict_over_time.png' ):       
        fig, axs = plt.subplots(self.n_ids, 1, figsize=(5* self.n_ids, 18))
        # Iterate over each station and plot in a subplot
        for ax, (station, models) in zip(axs, self.dict_datasets_by_id.items()):

            grt = models[1]['y_true']
            x = n_time_steps  # Number of time steps to plot

            for n in range(len(list(self.datasets_by_model.keys()))):
                model = models[n]['y_pred']
                ax.plot(np.arange(x), model[:x], label=list(self.datasets_by_model.keys())[n])

            ax.plot(np.arange(x), grt[:x], label="GRT")
            ax.set_xlabel('time steps')
            ax.set_ylabel(y_label)
            ax.legend(fontsize = 'large')
            ax.set_ylim(0,95)
            ax.grid(color='blue', linestyle='-.', alpha = 0.15, linewidth=0.7)
            ax.set_title(f"ID monitoring station: {station}", size= 15, loc = 'left')

        fig.suptitle(title, fontsize=21)
        plt.tight_layout()
        if save:
            plt.savefig(path, dpi = 250)
        plt.show()
     
    def get_metrics(self, path = 'metrics.csv',  save = False):

        for n, (model_name, model) in enumerate(self.datasets_by_model.items()):
            list_acc ={}    
        
            for station in self.ids:     
                current_df = model[model['id'] == station]
                list_acc[f'{model_name}_{station}'] = self.cal_acc(y_grt= current_df['y_true'], y_prd= current_df['y_pred'])
            array_acc = np.array(list(list_acc.values()))
            list_mean = array_acc.mean(axis=0)
            list_acc[f'{model_name}_mean'] = [ '%.3f' % elem for elem in list_mean]

            
            if n == 0:
                main_metrics = pd.DataFrame(list_acc).transpose()
            else:
                main_metrics = pd.concat([main_metrics, pd.DataFrame(list_acc).transpose()], axis= 0)
                

        
        main_metrics.columns= ["MAE", "RMSE", "MAPE", "MDAPE",  "R2"]
        if save:
            main_metrics.to_csv(path)
        return main_metrics

    def plot_residual_plot(self, path = 'residual_distribution_all_models.png', save = False):
        # Create a 2x3 grid of subplot figures
        fig, axs = plt.subplots(self.n_models, self.n_ids, figsize=(5* self.n_ids, 5 * self.n_models))
        fig.tight_layout(pad=5.0)  # Add space between plots

        # Flatten the array of axes for easy iteration
        axs = axs.flatten()

        # Define colormap and normalization
        cmap = plt.get_cmap('viridis')
        norm = Normalize()

        # Iterate over models and axes simultaneously
        for ax, (model_name, model_data) in zip(axs, self.dict_datasets_all.items()):
            x = model_data['y_true']
            y = model_data['y_true']- model_data['y_pred'] 
            z = np.histogram2d(x, y, bins=50, range=[[np.min(x), np.max(x)], [np.min(y), np.max(y)]])[0]
            z = np.clip(z, 0, np.max(z))  # Clip to avoid logarithm of zero
            scatter = ax.scatter(x, y, c=z[np.digitize(x, bins=np.linspace(np.min(x), np.max(x), 50)) - 1,
                                   np.digitize(y, bins=np.linspace(np.min(y), np.max(y), 50)) - 1],
                         cmap=cmap, norm=norm, alpha=0.4,  s=10)
            ax.set_title(f'Residual Plot for {model_name}')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Residuals')
            ax.set_xlim(0,100)
            ax.set_ylim(-50,60)
            ax.grid(color='blue', linestyle='-.', alpha = 0.15, linewidth=0.7)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # Create ScalarMappable and add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical')
        cbar.set_label('Density')
        if save:
            plt.savefig(path, dpi = 160)
        plt.show()



