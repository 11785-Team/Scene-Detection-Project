from opts import get_opts
import torch
import numpy as np
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim
from sklearn.model_selection import KFold
from dataloader import *
from train import Train
from generate import *
from detect import Detect
from metric import *

def main():
    opts = get_opts()
    # save filepath to one csv file
    train_path = opts.train_folder
    val_path = opts.val_folder

    print("train folder:", train_path)
    print("val folder:", val_path)

    train_csv_name = 'train_data.csv'
    save2csv(path=train_path, csvname=train_csv_name)
    val_csv_name = 'val_data.csv'
    # save2csv(path=val_path, csvname=val_csv_name)
    #csv_name = 'anime_data.csv'
    #save2csv(path=img_path, csvname=csv_name)
    
    # Can add some transform here

    # Define beta-vae net
    print('latent dim:', opts.latent_dim)
    Model = BetaVAE(in_channels=3, latent_dim=opts.latent_dim, hidden_dims=opts.hidden_dims, beta=opts.beta,
        gamma=opts.gamma, max_capacity=opts.max_capacity, Capacity_max_iter=opts.Capacity_max_iter, loss_type=opts.loss_type, tau=opts.tau)

    # model_state_path = '/content/gdrive/MyDrive/models_baseline_beta_1/model_state_4_val_loss_443.9217264811198.pkl'
    # Model.load_state_dict(torch.load(model_state_path))
    # print("continue training with " + model_state_path)

    # KFold training
    #K = 5   # split data into K parts
    #dataset = Dataload(imgpath=img_path, csv_name=csv_name)
    #kf = KFold(n_splits=K, shuffle=True)
    #train_loss_sum, val_loss_sum = 0, 0
    train_dataset = Dataload(imgpath=train_path, csv_name=train_csv_name)
    model_state = None
    print("Start Training!!!!!!!")

    model_state, train_loss, val_loss = Train(Model, train_dataset, None, batch_size=opts.bs,
    max_iters=opts.max_iters, lr=opts.lr, w_decay=opts.w_decay, m=opts.m, output_folder = opts.output_folder)
        

    torch.save(model_state, 'model_state.pkl')
    # use trained model to get the latent code of each frame
    # latent = generate_code(Model, model_state, dataset, opts.latent_dim)
    # save latent
    # np.save('latent.npy', latent)
    
    # define the distance metric for latent space
    metric = torch.nn.CrossEntropyLoss
    index_detected = Detect(latent, dis_metric=eud_dis)
    print("Frames at the following index experience scene change:")
    print(sorted(index_detected))


if __name__ == '__main__':
    main()

