# define different decoder structures
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
import numpy as np

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, key, query, mask=None):
        """_summary_

        Args:
            key (_type_): tensor([1,n_station,d_dim])
            query (_type_): tensor([1,n_station,d_dim])
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        n_station = key.shape[1]
        query = query.unsqueeze(1)
        score = torch.bmm(query, key.transpose(1, 2))
        if mask is not None:
            mask = mask.squeeze()
            score = score.masked_fill(mask == 0, -math.inf)
        attn = self.softmax(score.view(-1, n_station))
        return attn

class Local_Global_Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft,
        n_layers_rnn=1,
        rnn="GRU",
        cnn_hid_dim=128,
        fc_hid_dim=64,
        n_features=13,
        num_input_stat=10,
        n_layer = 2,
        activation = 'relu'
    ):
        super(Local_Global_Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        self.num_input_stat = num_input_stat - 1
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim*3, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft) # fc = 64, out 1
        self.linear3 = nn.Linear(fc_hid_dim, fc_hid_dim) 
        self.n_layer = n_layer
        self.fc_hid_dim = fc_hid_dim

        if activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'swish':
            self.act = nn.Hardswish()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)
        self.query_local = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key_local = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.value_local = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.atten = DotProductAttention()

        self.query = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key = nn.Linear(cnn_hid_dim, cnn_hid_dim)
        self.value = nn.Linear(cnn_hid_dim, cnn_hid_dim)
    
    def forward(self, x, h, l, climate):
        """
            x.shape = batch_size x (n1-1) x num_ft
            h.shape = batch_size x (n1-1) x latent_dim
            l.shape = (n1-1) x 1 = (27* 1)
        """
        x = x[:, -1, :, :]
        h = h[:, -1, :, :]
        l_ = l.unsqueeze(2)
        x_ = torch.cat((x, h), dim=-1)  # timestep x nodes x hidden feat
        ret = self.act(self.fc(x_))
        ret_ = ret.permute(0, 2, 1) # ret_ [batch, hidden_ft, n_node]
        interpolation_ = torch.bmm(ret_, l_)
        interpolation_ = interpolation_.reshape(ret.shape[0], -1)
        embed = self.embed(climate)

        query_local = self.query_local(torch.cat((interpolation_, embed), dim=-1))
        value_local = self.value_local(ret_)
        key_local = self.key_local(ret_)
        atten_weight_local = self.atten(key_local, query_local)
        atten_vector_local = torch.bmm(atten_weight_local.unsqueeze(1), value_local).squeeze()

        query = self.query(torch.cat((interpolation_, embed), dim=-1))
        value = self.value(ret)
        key = self.key(ret)
        atten_weight = self.atten(key, query)
        atten_vector = torch.bmm(atten_weight.unsqueeze(1), value).squeeze()     

        ret = self.linear(torch.cat((atten_vector_local,atten_vector, embed), dim=-1))  # (128, 1)
                
        if self.n_layer == 3:
            ret = self.act(ret)
            ret = self.linear3(ret) 
            ret = self.act(ret) 
            ret = self.linear2(ret) 
            return ret
        
        ret = self.act(ret)
        ret = self.linear2(ret) 
        return ret
    
def train_atten_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device):
    # wandb.watch(decoder, criterion, log="all", log_freq=100)
    decoder.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        batch_loss = 0
        y_grt = data["Y"].to(device).float()
        x = data["X"].to(device).float()
        G = data["G"][:,:,:,:,0].to(device).float()
        l = data["l"].to(device).float()
        cli = data['climate'].to(device).float()
        h = stdgi.embedd(x, G)
        y_prd = decoder(x, h, l,cli)  # 3x1x1
        batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    
    return epoch_loss / len(dataloader)

def test_atten_decoder_fn(stdgi, decoder, dataloader, device,criterion, scaler=None,test=True, args=None):
    decoder.eval()
    stdgi.eval()
    
    list_prd = []
    list_grt = []
    # breakpoint()
    epoch_loss = 0
    with torch.no_grad():
        for data in dataloader:
            batch_loss = 0
            y_grt = data["Y"].to(device).float()
            x = data["X"].to(device).float()
            G = data["G"][:,:,:,:,0].to(device).float()
            l = data["l"].to(device).float()
            cli = data['climate'].to(device).float()
            h = stdgi.embedd(x, G)
            y_prd = decoder(x, h, l,cli) 
            batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
            y_prd = torch.squeeze(y_prd).cpu().detach().numpy().tolist()
            y_grt = torch.squeeze(y_grt).cpu().detach().numpy().tolist()
            list_prd += y_prd
            list_grt += y_grt
            epoch_loss += batch_loss.item()

    a_max = scaler.data_max_[0]
    a_min = scaler.data_min_[0]
    list_grt = (np.array(list_grt) + 1) / 2 * (a_max - a_min) + a_min
    list_prd = (np.array(list_prd) + 1) / 2 * (a_max - a_min) + a_min
    list_grt_ = [float(i) for i in list_grt]
    list_prd_ = [float(i) for i in list_prd]
    if np.isnan(np.array(list_prd_)).any():
        mae = np.inf
    else:
        mae = mean_absolute_error(list_grt_, list_prd_)

    if test:
        return list_prd_, list_grt_, epoch_loss / len(dataloader)
    else:
        return mae , epoch_loss / len(dataloader)