import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='11785_Project')

    parser.add_argument('--max_iters', type=int, default=30, help='the number of epochs for training')

    # hyperparameters for optimizer
    parser.add_argument('--lr', type=float, default=0.1, help='the learning rate for training')
    parser.add_argument('--w_decay', type=float, default=5e-4, help='weight decay for optimizer')
    parser.add_argument('--m', type=float, default=0.5, help='momentum for optimizer')

    parser.add_argument('--bs', type=int, default=256, help='batch size')

    parser.add_argument('--threshold', type=float, default=0.5, help='the threshold for loss. Below it, training stops')

    # hyperparameters for beta-vae
    parser.add_argument('--latent_dim', type=int, default=1024, help='the dimension of latent space')
    parser.add_argument('--beta', type=int, default=4, help='beta value for beta-vae')
    parser.add_argument('--gamma', type=float, default=1000.0, help='gamma value for modified beta-vae')
    parser.add_argument('--max_capacity', type=int, default=25)
    parser.add_argument('--Capacity_max_iter', type=int, default=1e5)
    parser.add_argument('--loss_type', type=str, default='B')
    parser.add_argument('--hidden_dims', type=list, default=[32, 64, 128, 256, 512])

    opts = parser.parse_args()

    return opts