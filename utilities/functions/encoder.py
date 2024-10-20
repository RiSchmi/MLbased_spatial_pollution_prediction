import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

class GCN(nn.Module):
    '''
    initial representation as Graph Convolutional Network (GCN) layer that aggregates neighboring
    features based on their relevance, which is determined by the adjacency matrix.
    '''
    def __init__(self, infea, outfea, act="relu", bias=True):
        super(GCN, self).__init__()
        # Fully connected layer for transforming node features.
        self.fc = nn.Linear(infea, outfea, bias=False) # fc as weights of linear transformation
        self.act = nn.ReLU() if act == "relu" else nn.ReLU() # relu for non-linearity but computational simplicity

        # Optional bias for the layer.
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outfea))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        # Initialize weights using Xavier uniform distribution.
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        # neu la lop fully connectedd
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        '''
        args:
        * adj: adjacency matrix with edge relations (distance similarity)
        '''
        seq_fts = self.fc(seq) # Apply linear transformation
        if sparse: # Sparse matrix multiplication if the adjacency matrix is sparse.
            out = torch.unsqueeze(
                torch.bmm(adj, torch.squeeze(adj, torch.squeeze(seq_fts, 0)))
            )
        else:
            # Perform batch matrix-matrix product of the adjacency and feature matrices.
            out = torch.bmm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        return self.act(out)

class GCN_2_layers(torch.nn.Module):
    def __init__(self, hid_ft1, hid_ft2, out_ft, act='relu') -> None:
        super(GCN_2_layers, self).__init__()
        self.gcn_1 = GCN(hid_ft1, hid_ft2, act)
        self.gcn_2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj, sparse=False):
        
        x = self.gcn_1(x, adj)
        x  = self.gcn_2(x, adj)
        return x 

class TemporalGCN(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        
    """

    def __init__(self, device, in_channels, out_channels, hidden_dim, batch_size =1):
        
        super(TemporalGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self._create_parameters_and_layers()
        self.device = device

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(self.batch_size,X.shape[1], self.out_channels).to(X.self.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, adj, H):
        # import pdb; pdb.set_trace()
        h = self.conv_z(X, adj)
        Z = torch.cat([h, H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, adj, H):

        conv = self.conv_r(X, adj)
        R = torch.cat([conv, H], axis=2) # (b, 207, 64)
        # import pdb;pdb.set_trace()
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, adj, H, R):
        H_tilde = torch.cat([self.conv_h(X, adj), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, adj: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, adj, H)
        R = self._calculate_reset_gate(X, adj, H)
        H_tilde = self._calculate_candidate_state(X, adj, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H

class Attention_Encoder(nn.Module):
    """
    An encoder that applies an initial transformation to the node features and
    processes these features through a TemporalGCN to capture both spatial and
    temporal dynamics.
    """

    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, device):
        super(Attention_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.device = device
        self.rnn_gcn = TemporalGCN(in_channels= hid_ft1, out_channels= out_ft, hidden_dim= hid_ft2, device = self.device)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x))
        raw_shape = x.shape
        h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device= self.device) 
        
        list_h = []
        for i in range(raw_shape[1]):
            x_i = x[:,i,:,:].squeeze(1)  
            e = adj[:,i,:,:].squeeze(1) 
            h = self.rnn_gcn(x_i, e, h)
            list_h.append(h)
        h_ = torch.stack(list_h, dim=1)
        return h_

class Discriminator(nn.Module):
    def __init__(self, h_ft, x_ft, hid_ft):
        super(Discriminator, self).__init__()
        self.fc = nn.Bilinear(h_ft, x_ft, out_features=hid_ft)
        self.linear = nn.Linear(hid_ft, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, h, x, x_c):
        ret1 = self.relu(self.fc(h, x)) 
        ret1 = self.linear(ret1)
        ret2 = self.relu(self.fc(h, x_c))
        ret2 = self.linear(ret2)
        ret = torch.cat((ret1, ret2), 2)
        return self.sigmoid(ret)

class Attention_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, device, stdgi_noise_min=0.4, stdgi_noise_max=0.7):
        super(Attention_STDGI, self).__init__()
        self.encoder = Attention_Encoder(in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, device= device)
        
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max 

    def forward(self, x, x_k, adj): #x & x_k = node attributes
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        ret = self.disc(h[:,-1,:,:], x_k[:,-1,:,:], x_c[:,-1,:,:])
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts
        
    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h


def train_atten_stdgi(stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2):
    '''
    Trains the Attention STDGI model.
    Args:
        stdgi: The model to be trained.
        dataloader: DataLoader providing batches of data.
        optim_e: Optimizer for the encoder.
        optim_d: Optimizer for the discriminator.
        criterion: Loss function to measure the error between predictions and targets.
        device: Device (CPU/GPU) to perform computations.
        n_steps: Number of steps to train discriminator for each generator step.
    '''
    epoch_loss = 0
    stdgi.train() # Set the model to training mode.
    for data in (dataloader):#tqdm(dataloader): 
        for _ in range(n_steps):
            optim_d.zero_grad() # Reset gradients of discriminator optimizer
            d_loss = 0
            x = data["X"].to(device).float() # Input features
            G = data["G"][:,:,:,:,0].to(device).float() # Adjacency matrix  
            output = stdgi(x, x, G) # Forward pass through the model
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device) # True labels for discriminator
            d_loss = criterion(output, lbl) # Compute loss
            d_loss.backward() # Compute gradients
            optim_d.step() # Update discriminator parameters

        optim_e.zero_grad() # Reset gradients of encoder optimizer
        x = data["X"].to(device).float()
        G = data["G"][:,:,:,:,0].to(device).float()  
        output = stdgi(x, x, G)
        lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
        lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
        lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
        e_loss = criterion(output, lbl) # Calculate encoder loss
        e_loss.backward() # Compute gradients
        optim_e.step() # Update encoder parameters
        epoch_loss += e_loss.detach().cpu().item() # Accumulate loss
        
    return epoch_loss / len(dataloader)
