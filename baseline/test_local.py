from opts import get_opts
import torch
import numpy as np
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim
from sklearn.model_selection import KFold
# from dataloader import *
from train import Train
from generate import *
from detect import Detect
from metric import *



Model = BetaVAE(in_channels=3, latent_dim=1024, hidden_dims=[64, 128, 256, 512])
torch.save(Model.state_dict(), "toy_model.pth")

