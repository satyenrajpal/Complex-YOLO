import datetime
import os
import argparse

from solver import Solver



if __name__ == "__main__":
    
    # Parse args
    parser = argparse.ArgumentParser(prog = 'Train the mode.')
    parser.add_argument('--batch_size', type = int, default = 12, help = 'Number of images per batch.')
    parser.add_argument('--do_log', type = bool, default = True, help = 'Whether or not do logging.')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Initial learning rate.')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum coefficient.')
    parser.add_argument('--weight_decay', type = float, default = 0.0005, help = 'Weight decay.')
    parser.add_argument('--epochs', type = int, default = 1000, help = 'Number of epochs.')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/kitti_data/')
    parser.add_argument('--split', type = float, default = 0.9, help = 'Train/Val split')
    
    args = parser.parse_args()

    sol = Solver(args)
    sol.train()


