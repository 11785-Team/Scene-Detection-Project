import torch
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim
import os

def Train(model, train_dataset, val_dataset, batch_size, max_iters, lr, w_decay, m,):

    model_output_folder = '/content/gdrive/MyDrive/models_4_7_sleep/'
    if not os.path.exists(model_output_folder):
        os.mkdir(model_output_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = 2 if device != 'cpu' else 0
    print("Running Device is", device, " with num of workers ", nw)

    train_data_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nw)
    val_data_loader = data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=nw)

    # initialize model
    Model = model
    Model.to(device)

    # Optimzier
    # optimizer = optim.SGD(Model.parameters(), lr=lr, weight_decay=w_decay, momentum=m)
    optimizer = optim.Adam(Model.parameters(), lr=lr, weight_decay=w_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)
    # save training loss and validation loss for KFold
    train_loss = []
    val_loss = []

    # Training
    for iter in range(max_iters):
        Model.train()
        loss = 0
        recons_loss = 0
        kld_loss = 0
        batch_num = 0
        for images in train_data_loader:
            imgs = images.to(device)

            optimizer.zero_grad()
            output = Model.forward(imgs)

            loss_tmp = Model.loss_function(output, M_N=batch_size)
            
            # keep records of recons loss and kld loss as well
            loss += loss_tmp['loss'].item()
            recons_loss += loss_tmp['Reconstruction_loss'].item()
            kld_loss += loss_tmp['KLD'].item()
            # print('total:', loss_tmp['loss'].item(), 'kld loss:', loss_tmp['KLD'].item(), 'recons loss:', loss_tmp['Reconstruction_loss'].item())
            loss_tmp['loss'].backward()
            optimizer.step()
            # compute the number of batch
            batch_num += 1
            torch.cuda.empty_cache()
            del imgs

        print('Train: #{} epoch, the loss is {}, recons loss is {}, kld loss is {}'.format(iter, 
                                                        loss/batch_num, recons_loss/batch_num, kld_loss/batch_num))
        train_loss.append(loss / batch_num)

        # Validation
        batch_num = 0
        loss_val = 0
        Model.eval()
        with torch.no_grad():
            for images in val_data_loader:
                imgs = images.to(device)
                output = Model.forward(imgs)
                loss_val += Model.loss_function(output, M_N=batch_size)['loss'].item()
                batch_num += 1
                torch.cuda.empty_cache()
                del imgs

        print('Val: #{} epoch, the loss is'.format(iter), loss_val / batch_num)
        val_loss.append(loss_val / batch_num)
        # save model for every 5 epoch to gdrive
        scheduler.step(loss_val/batch_num)
        # if iter % 5 == 4:
        output_model_path = os.path.join(model_output_folder, 'model_state_' + str(iter) + '_val_loss_' + str(loss_val/batch_num) + '.pkl')
        torch.save(Model.state_dict(), output_model_path)
    return Model.state_dict(), train_loss, val_loss






