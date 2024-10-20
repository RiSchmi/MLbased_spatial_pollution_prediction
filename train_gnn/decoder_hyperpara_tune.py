from functions.constructor import Preprocessing, AQDataSet # load own external functions
from functions.encoder import Attention_STDGI
from functions.decoder import Local_Global_Decoder,train_atten_decoder_fn, test_atten_decoder_fn
from functions.utilities import EarlyStopping
from torch.utils.data import DataLoader
import torch, os
import torch.nn as nn
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import time
import numpy as np


# 1. define device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. construct data array
main = Preprocessing(path_data= os.path.abspath('data/datasets/df_processed.csv'),
                     path_coor= os.path.abspath('data/monitoring_station/monitoring_station.shp'))

stations = main.stations
distance_matrix = main.get_distance_matrix()
corr, norm_loc, norm_temp, norm_imputed, scaler = main.construct_data()


sequence_length = 12
mse_loss = nn.MSELoss()
train_station = [2, 4, 5, 6, 8, 9, 11, 12, 14, 15] + [0, 3, 13]

def objective(space):
    print(space)

    start_time = time.time()

    # adjustable parameter decoder architecture:
    cnn_hid_dim = int(space['cnn_units']) # 64 # hidden neuron in cnn
    fc_hidden_dim = int(space['dense_units']) # 128 # units in dense layer
    activation = (space['act_decoder'])
    n_layer= space['n_layer']

        # hyperparamter optimizer:
    optimizer = (space['optimizer'])
    l2_coef = (space['l2']) # 0 # l2 regularization 
    lr_decoder = (space['lr']) # 0.001 # learning rate 
    momentum = (space['momentum'])# 0.9 # momentum for stochastic gradient descent
    batch_size = int(space['batchsize'])
    
        # initialize decoder

    decoder = Local_Global_Decoder(norm_loc.shape[2] + 60,
                            1,
                            n_layers_rnn= 1,
                            cnn_hid_dim= cnn_hid_dim, # adjustable
                            fc_hid_dim= fc_hidden_dim, # adjustable
                            n_features= norm_temp.shape[2],  
                            num_input_stat= len(train_station) -1,
                            n_layer= n_layer,
                            activation= activation
                            ).to(device)
    
    if optimizer == 'adam':
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr = lr_decoder, weight_decay = l2_coef)
    elif optimizer == 'sgd':
        optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr = lr_decoder, momentum = momentum, weight_decay= l2_coef)
    elif optimizer == 'rmsprop':
        optimizer_decoder = torch.optim.RMSprop(decoder.parameters(), lr = lr_decoder, alpha=0.99, weight_decay=l2_coef, momentum= momentum)

    total_loss = 0
    mae_loss = 0

    for current_val_station in train_station:
        current_train_stations = train_station.copy()
        current_train_stations.remove(current_val_station)

        # initialize new model 
        stdgi = Attention_STDGI(
            in_ft=norm_loc.shape[2], out_ft=60, en_hid1=48, en_hid2=64, dis_hid=6,
            stdgi_noise_min=0.4, stdgi_noise_max=0.7, device= device)


        # load the state dictionary of embedding
        checkpoint = torch.load(f'output/encoder/station_{current_val_station}.pt', map_location= device)
        stdgi.load_state_dict(checkpoint['model_dict'])
        stdgi.to(device)

        

        # specify training data
        train_dataset = AQDataSet(
            data_df= norm_loc,  
            climate_df= norm_temp,
            distance_matrix= distance_matrix,
            list_train_station= current_train_stations,
            input_dim= sequence_length,
            corr=corr        
        )

        
        train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=0)


        # specify early stopping
        early_stopping_decoder = EarlyStopping(patience= 8, 
                                            verbose= False, 
                                            delta= 0.001)



        # train the model under current setup
        min_mae = 99999
        min_loss= 99999 
        
        for i in range(1):

          

            if not early_stopping_decoder.early_stop:
                train_loss = train_atten_decoder_fn(
                    stdgi,
                    decoder,
                    train_dataloader,
                    mse_loss,
                    optimizer_decoder,
                    device,
                )
                
                valid_loss = 0
                mae = 0

                
                valid_dataset = AQDataSet(
                    data_df=norm_loc, 
                    climate_df=norm_temp,
                    distance_matrix = distance_matrix,
                    list_train_station=current_train_stations,
                    test_station= current_val_station,
                    valid=True,
                    input_dim=12,
                    corr=corr,
                )

                valid_dataloader = DataLoader(valid_dataset, batch_size= batch_size, shuffle=False, num_workers=0)
                
                mae , valid_loss = test_atten_decoder_fn(
                    stdgi,
                    decoder,
                    valid_dataloader,
                    device,
                    mse_loss,
                    scaler= scaler,
                    test=False,
                                )
                
                early_stopping_decoder(valid_loss, decoder)

                if mae < min_mae:
                    min_mae = mae
                
                if valid_loss < min_loss:
                    min_loss = valid_loss
                if i%2== 0:
                    print(f"training loss: {train_loss}, validation loss {valid_loss}")


        total_loss += min_loss
        mae_loss += min_mae
        print(f'current model finished with mae: {min_mae}')
    
    average_mae = mae_loss / len(train_station)
    average_loss = total_loss/ len(train_station)
    print(f"Valid loss: {average_loss} / MAE (val): {average_mae}")
    print(space)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {'loss': average_loss, 'status': STATUS_OK }


space ={'l2': hp.uniform("x_l2", 0, 0.02),
        'lr': hp.loguniform ('x_lr', np.log(0.0005), np.log(0.003)),
        'cnn_units': hp.quniform ('x_cnn', 32, 128, 16),
        'dense_units' : hp.quniform ('x_dense', 32, 128, 16),
        'momentum' : hp.uniform('x_momentum', 0.5, 0.99), 
        'optimizer': hp.choice('x_optimizer', ['adam', 'sgd', 'rmsprop']),
        'act_decoder': hp.choice('x_act_decoder', ['relu', 'swish']),
        'n_layer': hp.choice('x_n_layer', [2, 3]),
        'batchsize': hp.quniform ('x_batch', 16, 48, 8),        
    }

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals= 70,
            trials= Trials())

print(best)
