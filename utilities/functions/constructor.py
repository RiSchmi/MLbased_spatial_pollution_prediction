import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class Preprocessing: # class for location and feature based preprocessing and transformations

    def __init__(self, path_data, path_coor, path_impute_index): 
    
    # construct features and main data frame   
        self.features_target = ['NO2'] 
        
        self.features_location = ['prop_intercept_200', 'prop_intercept_50', 'gvi_50', 'gvi_200', 'tvi_50', 
                                  'tvi_200', 'prop_main_emitter_200', 'distance_nearest_street', 'distance_nearest_intersection', 'population_200','population_50','free_wind'] # len 11
        
        self.features_temporal = ['weekend', 'rushhour', 'lai_factor', 'prec_mm', 'prec_bool', 'humidity',
                                  'temp', 'radiation', 'wind_speed', 'air_pressure', 'wind_degree',] # len 12 
        
        self.index_split = len(self.features_location) + 1
        
        # construct target & feature data as df
        self.data = pd.read_csv(path_data)
        self.data = self.data[['id', 'time_step'] + self.features_target + self.features_temporal + self.features_location]
        self.split_point = 7883 # threshold for test instances (10 percent)/ last 10 percent
        self.scaler = MinMaxScaler((-1,1))
        
        # construct station data
        self.coor = gpd.read_file(path_coor)[['id','stattyp', 'geometry']]
        self.coor['id'] = self.coor['id'].apply(lambda x : x.lower().replace(' ', '')[:5]) # adjust name of the station id
        self.coor = self.coor.to_crs(epsg=4326) # convert reference system
        self.coor['latitude'] = self.coor['geometry'].x # Extract longitude as separate column
        self.coor['longitude'] = self.coor['geometry'].y # Extract latitude as separate column

        # path to dictonary with index of imputed values (later excluded for test set)
        self.path_na_values = path_impute_index

        # define stations
        self.stations = np.array(self.data['id'].unique())

        # define centroid (TV Tower Berlin) for distance similarity matrix
        self.lat = 13.40945
        self.long = 52.520803

# 1. processing data 

    def construct_data(self):
        ''' transform target, feature dataframe to normalized & splitted data array of shape: [n_observation, n_sites, n_features]
        -----
        Return:
        * corr (np.array (n(stations), n(stations)) of corelation matrix between target feature (NO2)
        * train_loc (np.array (7.883, 16, 12)), normalized train set with 90 % of instances of location related features 
        * train_temp (np.array (7.883, 16, 12)), norm. train set with 90 % of instances of temporal related features 
        * test_loc (np.array (876, 16, 12)), norm. test with 10 % of instances of location related features 
        * test_temp (np.array (876, 16, 12)), norm. test set with 10 % of instances of temporal related features 
        * scaler: to transform arrays back 
        '''
        def is_imputed(self):
            '''add bool if value is imputed'''
            with open(self.path_na_values, 'rb') as pickle_file:
                timestep_missing_data = pickle.load(pickle_file) # load data with index of imputed values

            df = self.data
            df['is_imputed'] = df.apply(lambda row: row['time_step'] in timestep_missing_data.get(row['id'], []), axis=1)
            return df

        def construct_data_array(self, df): 
            
            list_arr = [] # initiate list for arrays with per site data
            for id in self.stations: # iterate over all stations
                current_df = df[df['id'] == id].drop(['id', 'time_step'], axis = 1) # filter data frame by id
                array = np.expand_dims(current_df, axis=1) # convert to numpy array
                list_arr.append(array) # append current array
                
            data_array = np.concatenate(list_arr, axis=1) # .shape = (8759, 16, 24) as [n_observation, n_sites, n_features]
            return data_array
                   
        def test_training_split(self, data):
            
            train_data = data[:self.split_point]
            test_data = data[self.split_point:]
            return test_data, train_data         

        def rescale_data(self, data, idx_temp):
            
            # set sclaer
            scaler = self.scaler

            # Reshape and scale the training data
            train_shape = data.shape
            train_reshaped = np.reshape(data, (-1, train_shape[2]))
            train_scaled = self.scaler.fit_transform(train_reshaped)
            train_scaled = np.reshape(train_scaled, train_shape)

            # Split training data into features and target (assuming target is at idx_temp)
            train_loc = train_scaled[:, :, :idx_temp]
            train_temp = train_scaled[:, :, idx_temp:-1]
            train_imputed = train_scaled[:, :, -1:]

            # Reshape and scale the test data using the same scaler
            # test_shape = test_data.shape
            # test_reshaped = np.reshape(test_data, (-1, test_shape[2]))
            # test_scaled = self.scaler.transform(test_reshaped)
            # test_scaled = np.reshape(test_scaled, test_shape)

            # # Split test data into features and target
            # test_loc = test_scaled[:, :, :idx_temp]
            # test_temp = test_scaled[:, :, idx_temp:-1]
            # test_imputed = test_scaled[:, :, -1:]

            return train_loc, train_temp, train_imputed, scaler

               

        def get_corr_matrix(data):
            # create correlation matrix for target variable between station 
            no2 = data[:,:,0]
            corr = pd.DataFrame(no2).astype('float').corr().values
            return corr
    
        
        data_array = construct_data_array(self, df = is_imputed(self)) # define data array .shape: .shape = (8759, 16, 24)
        corr = get_corr_matrix(data = data_array) # define correlation_matrix              
        #test_data, train_data = test_training_split(self, data = data_array) # define test- training split
        #train_loc, train_temp, train_imputed, test_loc, test_temp, test_imputed, scaler = rescale_data(self, test_data, train_data, idx_temp = 12)

        
        norm_loc, norm_temp, norm_imputed, scaler = rescale_data(self, data_array, idx_temp = 12)
        return corr, norm_loc, norm_temp, norm_imputed, scaler
        
    
# 2. processing coordinates:
    def get_distance_matrix(self):
        '''
        returns matrix (n(stations), n(stations)) of difference in distance to centroid point 
        --------
        * centroid: (defined in __init__)
        '''

        def distance_center(self, point1):
            '''calculate distance from point to centroid
            -------
            * longitude, latitude = coordinates of defined centroid
            * point1: long, lat array with shape: (2,)
            * RETURN: distance between points in km
            '''
            center_point = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[Point(self.long, self.lat)])
            center_point = center_point.to_crs('EPSG:25833').geometry[0]

            point = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[Point(point1[0], point1[1])])    
            point = point.to_crs('EPSG:25833').geometry[0]

            return center_point.distance(point) # in km

        def precompute_distances(self, locations):
            '''creates difference in distance to centroid matrix
            ----------
            RETURN: look-up table .shape: (n(stations), n(stations) '''
            num_locations = len(locations)
            distance_matrix = np.zeros((num_locations, num_locations))
            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j:
                        distance_i = distance_center(self, locations[i])
                        distance_j = distance_center(self, locations[j])

                        distance_matrix[i][j] = np.abs(distance_i - distance_j)
            return distance_matrix

        def get_coordinates(self):
            '''extract coordinates of stations in order of appearance
            ------
            RETURN:
            * location_ (np.array): coordinates with monitoring stations .shape (n(stations), 2)
            '''
            coor_list = [] # initiate list for arrays for per site coordinates
            for id in self.stations:
                coor_list.append(np.array(self.coor[self.coor['id'] == id][['longitude', 'latitude']]))
            
            return  np.concatenate(coor_list, axis=0) # location_: needed to construct distance matrix

        return precompute_distances(self, locations= get_coordinates(self))   # distance matrix

from torch.utils.data import Dataset
import random

class AQDataSet(Dataset): # creation of graph structure
    def __init__(
        self,
        data_df,
        climate_df,
        distance_matrix,
        list_train_station,
        input_dim,
        test_station=None,
        test=False,
        valid=False,
        corr=None,
        
    ) -> None:
        super().__init__()
        assert not (test and test_station == None)
        assert not (test_station in list_train_station)
        self.list_cols_train_int = list_train_station
        self.input_len = input_dim
        self.test = test
        self.valid = valid 
        self.data_df = data_df
        self.distance_matrix = distance_matrix
        self.climate_df = climate_df
        self.n_st = len(list_train_station) - 1
        self.corr =corr
        self.train_cpt = 0.6
        self.valid_cpt = 0.25
        self.test_cpt = 0.15
        

        idx_test = int(len(data_df) * (1- self.test_cpt))
        self.X_train = data_df[:idx_test,:, :]
        self.climate_train = climate_df[:idx_test,:, :]
        
        # test data
        if self.test:
            
            test_station = int(test_station)
            self.test_station = test_station
            lst_cols_input_test_int = list(set(self.list_cols_train_int) - set([self.list_cols_train_int[-1]])            )
            self.X_test = data_df[idx_test:, lst_cols_input_test_int,:]
            
            self.l_test = self.get_distance_similarity_matrix(
                lst_cols_input_test_int, test_station
            )
            self.Y_test = data_df[idx_test:, test_station, :]
            self.climate_test = climate_df[idx_test:, test_station, :]
            self.G_test = self.get_adjacency_matrix(lst_cols_input_test_int)
            if self.corr is not None:
                self.corr_matrix_test = self.get_corr_matrix(lst_cols_input_test_int)
        
        elif self.valid:
            # phan data test khong lien quan gi data train 
            test_station = int(test_station)
            self.test_station = test_station
            lst_cols_input_test_int = list(set(self.list_cols_train_int) - set([self.list_cols_train_int[-1]])            )
            self.X_test = data_df[:idx_test, lst_cols_input_test_int,:]

            self.l_test = self.get_distance_similarity_matrix(
                lst_cols_input_test_int, test_station
            )
            self.Y_test = data_df[:idx_test, test_station, :]
            self.climate_test = climate_df[:idx_test, test_station, :]
            self.G_test = self.get_adjacency_matrix(lst_cols_input_test_int)

            if self.corr is not None:
                self.corr_matrix_test = self.get_corr_matrix(lst_cols_input_test_int)


    def get_corr_matrix(self, list_station):
            corr_mtr = self.corr[np.ix_(list_station,list_station)]
            corr_mtr_ = np.expand_dims(corr_mtr.sum(-1),-1)
            corr_mtr_ = np.repeat(corr_mtr_,corr_mtr_.shape[0],-1)
            corr_mtr = corr_mtr/corr_mtr_
            corr_mtr = np.expand_dims(corr_mtr,0)
            corr_mtr = np.repeat(corr_mtr, self.input_len,0)
            return corr_mtr
      

    def get_distance_similarity_matrix(self, lst_col_train_int, target_station):
        '''
        create weight distribution based on similarity in distance
        ----------
        * lst_col_train_int: (list) index of stations to compare distance
        * target_station: (int) index of target station to compare
        * RETURN: (array) shape= (len(lst_col_train_int),) with distribution (sum = 1) of similarity 
        in distance as array(normalized([1/abs(distance(x) - distance(target))] for x in lst_col_train_int)
        '''
        matrix = []
        for i in lst_col_train_int:
            distance = self.distance_matrix[i][target_station]

            if distance == 0:
                matrix.append(0)
            else:
                matrix.append(1 / distance)

        matrix = np.array(matrix)
        matrix = matrix / matrix.sum()
        return matrix

    def get_adjacency_matrix(self, lst_col_train_int, target_station_int=None):
        
        adjacency_list = [] # initialize list for per site distance similarity
        for i in lst_col_train_int:
            adjacency_array = self.get_distance_similarity_matrix(lst_col_train_int, i)
            adjacency_list.append(adjacency_array)

        adjacency_matrix = np.vstack(adjacency_list)     
        adjacency_matrix = np.expand_dims(adjacency_matrix, 0)
        adjacency_matrix = np.repeat(adjacency_matrix, self.input_len, 0)
        
        return adjacency_matrix
        
    
    def __getitem__(self, index: int):
        list_G = []
        if self.test:
            x = self.X_test[index : index + self.input_len, :]
            y = self.Y_test[index + self.input_len - 1, 0]
            G = self.G_test
            l = self.l_test
            climate = self.climate_test[index + self.input_len - 1, :]
            if self.corr is not None:
                list_G = [G,self.corr_matrix_test]
            else: 
                list_G = [G]
        elif self.valid:
            x = self.X_test[index : index + self.input_len, :]
            y = self.Y_test[index + self.input_len - 1, 0]
            G = self.G_test
            l = self.l_test
            climate = self.climate_test[index + self.input_len - 1, :]
            if self.corr is not None:
                list_G = [G,self.corr_matrix_test]
            else: 
                list_G = [G]
        else:
            # remove one random sample from training list
            picked_target_station_int = random.choice(self.list_cols_train_int)
            lst_col_train_int = list(
                set(self.list_cols_train_int) - set([picked_target_station_int])
            )
            x = self.X_train[index : index + self.input_len, lst_col_train_int, :]
                      
            y = self.X_train[index + self.input_len - 1, picked_target_station_int, 0]
            climate = self.climate_train[
                index + self.input_len - 1, picked_target_station_int, :
            ]
            G = self.get_adjacency_matrix(
                lst_col_train_int, picked_target_station_int
            )
            if self.corr is not None:
                corr_matrix = self.get_corr_matrix(lst_col_train_int)
                list_G = [G,corr_matrix]
            else: 
                list_G = [G]

            l = self.get_distance_similarity_matrix(lst_col_train_int, picked_target_station_int)
            
        sample = {
            "X": x,
            "Y": np.array([y]),
            # "G": np.array(G),
            "l": np.array(l),
            "climate": climate,
        }
        sample["G"] = np.stack(list_G,-1)
        # breakpoint()
        return sample

    def __len__(self) -> int:
        if self.test:
            return self.X_test.shape[0] - self.input_len
        return self.X_train.shape[0] - (self.input_len)
