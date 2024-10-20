# import own moduls
from utilities.functions.constructor import Preprocessing, AQDataSet # to arrange data array + graph structure
from utilities.functions.encoder import Attention_STDGI, train_atten_stdgi
from utilities.functions.utilities import EarlyStopping

import time, torch, os
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials # moduls for hyperparameter tuning
from torch.utils.data import DataLoader # transform to es for data handling

# 1. define device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. construct data array
main = Preprocessing(path_data= os.path.abspath('data/df_2023_imputed.csv'),
                     path_coor= os.path.abspath('data/monitoring_station/monitoring_station.shp'),
                     path_impute_index= os.path.abspath('data/index_imputed_no2_values.pkl'))
                    

stations = main.stations
distance_matrix = main.get_distance_matrix()
corr, norm_loc, norm_temp, norm_imputed, scaler = main.construct_data()

# 3. defined parameter
bce_loss = torch.nn.BCELoss()
train_station = [0, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15]

# 4. objective function: iterates over leave one station out cross validation
def objective(space):
    start_time = time.time() # track time per LOOCV
    best_bce = []
    best_epochs = []
    # 4.1 define hyperparameter 
    absolute_distance = space['distance_type'] # absolute distance or difference in distance for adjacency
    

        # graph represention learning (encoder)
    batch_size = int(space['batch_size']) # 32 
    en_hid1 = int(space['en_hid1'])
    en_hid2 = int(space['en_hid2'])
        
        # parameter generator (encoder)
    opt_e = space['opt_e']
    lr_e = space['lr_e']
    l2_e = space['l2_e']
    momentum_e = space['momentum_e']
          
        # parameter discriminator
    opt_d = space['opt_d']
    lr_d = space['lr_d']
    l2_d = space['l2_d']
    momentum_d = space['momentum_d']


    

    # start leave one station out cross validation 
    for current_val_station in train_station:
        current_train_stations = train_station.copy()
        current_train_stations.remove(current_val_station)

        # 1. construct graph neural network for current training stations
        train_dataset = AQDataSet(
            data_df= norm_loc,  
            climate_df= norm_temp,
            distance_matrix= distance_matrix,
            list_train_station= current_train_stations,
            absolute_distance = absolute_distance,
            input_dim= 12, #sequence_length
            corr=corr)
        
        train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=0)

        # 2. initialize spatio-temporal deep graph infomax
        stdgi = Attention_STDGI(
            in_ft= norm_temp.shape[2],
            out_ft= 60,
            en_hid1= en_hid1, 
            en_hid2= en_hid2, 
            dis_hid= 6,
            device= device,
            stdgi_noise_min= 0.4, 
            stdgi_noise_max= 0.7, 
        ).to(device)

        # define optimizer (+ its parameter) for generator (encoder) and discriminator
        if opt_e == 'adam':    
            stdgi_optimizer_e = torch.optim.Adam(stdgi.encoder.parameters(), lr= lr_e, weight_decay=l2_e)
        elif opt_e == 'sgd':
            stdgi_optimizer_e = torch.optim.SGD(stdgi.encoder.parameters(), lr = lr_e, momentum = momentum_e, weight_decay= l2_e)
        elif opt_e == 'rmsprop':
            stdgi_optimizer_e = torch.optim.RMSprop(stdgi.encoder.parameters(), lr = lr_e, alpha=0.99, weight_decay=l2_e, momentum= momentum_e)
        
        if opt_d == 'adam':    
            stdgi_optimizer_d = torch.optim.Adam(stdgi.encoder.parameters(), lr= lr_d, weight_decay=l2_d)
        elif opt_d == 'sgd':
            stdgi_optimizer_d = torch.optim.SGD(stdgi.disc.parameters(), lr = lr_d, momentum = momentum_d, weight_decay= l2_d)
        elif opt_d == 'rmsprop':
            stdgi_optimizer_d = torch.optim.RMSprop(stdgi.disc.parameters(), lr = lr_d, alpha=0.99, weight_decay=l2_d, momentum= momentum_d)

        # 3. run model 
        early_stopping_stdgi = EarlyStopping(patience=4, delta=0.004)

        min_loss = 99999
        best_epoch = 0

        for i in range(20):
            if not early_stopping_stdgi.early_stop:
                loss = train_atten_stdgi(
                    stdgi,
                    train_dataloader,
                    stdgi_optimizer_e,
                    stdgi_optimizer_d,
                    bce_loss,
                    device,
                    n_steps=2, 
                )

                early_stopping_stdgi(loss, stdgi)
                
                if loss < min_loss:
                    min_loss = loss
                    best_epoch = i

                print(f"Epochs {i} Loss: {loss}")
        
        print(f"min binary cross entropy loss : {min_loss} after {best_epoch} epochs")
        best_bce.append(min_loss)
        best_epochs.append(best_epoch)

    average_loss = sum(best_bce)/len(best_bce)
    print(space)
    print(f'mean bce score: {average_loss} with {best_bce}')
    print(f'mean best epoch {sum(best_epochs)/len(best_epochs)} with {best_epochs}')
    print(f"--- {time.time() - start_time}s seconds for one iteration of LOOCV ---")
    return {'loss': average_loss, 'status': STATUS_OK }




space ={'batch_size': hp.quniform('x_batch', 26, 56, 8),
        'en_hid1': hp.quniform('x_en_hid1', 48, 112, 16),
        'en_hid2': hp.quniform('x_en_hid2', 48, 112, 16),      
        'opt_e': hp.choice('x_opt_e', ['adam', 'sgd', 'rmsprop']),
        'lr_e': hp.loguniform ('x_lr_e', np.log(0.0005), np.log(0.003)),
        'l2_e': hp.uniform("x_l2_e", 0, 0.02),
        'momentum_e':hp.uniform('x_momentum_e', 0.5, 0.99), 
        'opt_d': hp.choice('x_opt_d', ['adam', 'sgd', 'rmsprop']),
        'lr_d': hp.loguniform ('x_lr_d', np.log(0.0005), np.log(0.003)),
        'l2_d': hp.uniform("x_l2_d", 0, 0.02),
        'momentum_d':hp.uniform('x_momentum_d', 0.5, 0.99),
        'distance_type': hp.choice('x_distance_type', [True, False]),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals= 50,
            trials= Trials())

print(best)
