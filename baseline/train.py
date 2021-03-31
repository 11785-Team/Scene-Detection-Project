import torch
import torch.utils.data as data
from beta_vae import BetaVAE
import torch.optim as optim

def Train(model, train_dataset, val_dataset, batch_size, max_iters, lr, w_decay, m,):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = 8 if device != 'cpu' else 0
    print("Running Device is", device, " with num of workers ", nw)

    train_data_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nw)
    val_data_loader = data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=nw)

    # initialize model
    Model = model
    Model.to(device)

    # Optimzier
    optimizer = optim.SGD(Model.parameters(), lr=lr, weight_decay=w_decay, momentum=m)

    # save training loss and validation loss for KFold
    train_loss = []
    val_loss = []

    # Training
    for iter in range(max_iters):
        Model.train()
        loss = 0
        batch_num = 0
        for images in train_data_loader:
            imgs = images.to(device)

            optimizer.zero_grad()
            output = Model.forward(imgs)

            loss_tmp = Model.loss_function(output, M_N=batch_size)['loss']
            loss += loss_tmp.item()

            loss_tmp.backward()
            optimizer.step()
            # compute the number of batch
            batch_num += 1
            print(loss_tmp.item())
        print('Train: #{} epoch, the loss is'.format(iter), loss / batch_num)
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
        print('Val: #{} epoch, the loss is'.format(iter), loss_val / batch_num)
        val_loss.append(loss_val / batch_num)
    return Model.state_dict(), train_loss, val_loss






