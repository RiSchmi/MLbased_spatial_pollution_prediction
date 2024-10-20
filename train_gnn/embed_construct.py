import os 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utilities.functions.constructor import Preprocessing, AQDataSet # load own external functions
from utilities.functions.encoder import Attention_STDGI, train_atten_stdgi
from utilities.functions.utilities import EarlyStopping
import logging

# 1. define device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. construct data array
main = Preprocessing(path_data= os.path.abspath('data/df_2023_imputed.csv'),
                     path_coor= os.path.abspath('data/monitoring_station/monitoring_station.shp'),
                     path_impute_index= os.path.abspath('data/index_imputed_no2_values.pkl'))
                    

stations = main.stations
distance_matrix = main.get_distance_matrix()
corr, norm_loc, norm_temp, norm_imputed, scaler = main.construct_data()


# 3. train 13 graph embeddings
    # 3.1 adjustable hyperparameter

sequence_length = 12
batch_size = 24# 32 
lr_encoder = 0.001790 # 0.001 # adjustable 
lr_decoder = 0.000612 #0.001 # adjustable
l2_coef_e = 0.007949
l2_coef_d = 0.000203
momentum_d = 0.52711


# initiate leave one station out 
train_station = [2, 4, 5, 6, 8, 9, 11, 12, 14, 15] + [0, 3, 13]
bce_loss = nn.BCELoss()

for current_val_station in train_station:
    current_train_stations = train_station.copy()
    current_train_stations.remove(current_val_station)

    # 1. construct graph neural network for current training stations
    train_dataset = AQDataSet(
        data_df= norm_loc,  
        climate_df= norm_temp,
        distance_matrix= distance_matrix,
        list_train_station= current_train_stations,
        input_dim= sequence_length,
        corr=corr        
    )

    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=0)

    # 2. initialize network architecture
    stdgi = Attention_STDGI(
        in_ft= norm_loc.shape[2],
        out_ft= 60,
        en_hid1= 64, # adjustable
        en_hid2= 64, # adjustable
        dis_hid= 6,
        device= device,
        stdgi_noise_min= 0.4, # adjustable
        stdgi_noise_max= 0.7, # adjustable
    ).to(device)

    stdgi_optimizer_e = torch.optim.Adam(stdgi.encoder.parameters(), lr= lr_encoder, weight_decay=l2_coef_e)
    stdgi_optimizer_d = torch.optim.RMSprop(stdgi.disc.parameters(), momentum= momentum_d, lr= lr_decoder, weight_decay=l2_coef_d)

    # 3. run model 
    early_stopping_stdgi = EarlyStopping(patience=8, verbose=True, delta=0.003, path= os.path.abspath(f'output/encoder/station_{current_val_station}_local+temp.pt'))

    
    logging.info(f"Training stdgi || attention decoder || epochs {30} || lr {lr_encoder}")

    train_stdgi_loss = []
    for i in range(20):
        if not early_stopping_stdgi.early_stop:
            loss = train_atten_stdgi(
                stdgi,
                train_dataloader,
                stdgi_optimizer_e,
                stdgi_optimizer_d,
                bce_loss,
                device,
                n_steps=2, # adjustable
            )
            early_stopping_stdgi(loss, stdgi)
                    
            logging.info("Epochs/Loss: {}/ {}".format(i, loss))
            print(loss)
