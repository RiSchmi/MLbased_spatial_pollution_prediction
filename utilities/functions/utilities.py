import torch
import numpy as np 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=10, verbose=False, delta=0.003, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement before stopping the training.
            verbose (bool): If true, outputs messages to stdout.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Where to save the model when it improves.
            trace_func (function): Function used for logging progress.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Call method updates the status of early stopping by checking if the validation loss has decreased.
        """
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score + self.delta > self.best_score:
            self.counter += 1
            if self.counter %2 == 0:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            
            checkpoints = {"model_dict": model.state_dict()}
            torch.save(checkpoints, self.path)
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).")
    
        self.val_loss_min = val_loss