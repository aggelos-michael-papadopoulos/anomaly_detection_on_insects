from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Grayscale
import argparse
import torch.nn.functional as F
from torch.optim import Adam
import torch
from trainer import train
from model import AnomalyAE
from datetime import datetime
from Dataloader import CustomDataset


def create_datagen(data_dir, img_dir, batch_size=16):

    dataset = CustomDataset(img_dir)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',
                        default='/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/all data/TrainDir/',
                        help="Please specify the train directory")
    parser.add_argument('--val_dir',
                        default='/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/all data/ValDir/',
                        help="Please specify the test directory")
    parser.add_argument(
        "--log_interval", type=int, default=10,
        help="how many batches to wait before logging training status"
    )
    parser.add_argument('--epochs',
                        type=int,
                        default=35,                                                                 # 16
                        help="Please specify the number of epochs")
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=8,                                                                  # 4
                        help="Please specify the batch_size")
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=2,                                                                  # 4
                        help="Please specify the batch_size")

    parser.add_argument(
        "--log_dir", type=str,
        default=f'tensorboard_logs_{datetime.now().strftime("%d%m%Y_%H-%M")}',
        help="log directory for Tensorboard log output"
    )
    parser.add_argument(
        '--load_weight_path', type=str,
        help="Please specify the weight path that needs to be loaded.")

    parser.add_argument(
        '--save_graph', action='store_true',
        help="Specify this if you want to save the network graph.")

    args = parser.parse_args()

    optimizer = Adam
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = F.mse_loss
    train_loader = create_datagen(args.train_dir, "/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/all data/TrainDir/NonDefect/",  args.train_batch_size)
    val_loader = create_datagen(args.val_dir, "/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/all data/ValDir/NonDefect/", args.val_batch_size)
    model = AnomalyAE()

    train(model, optimizer, loss, train_loader,
          val_loader, args.log_dir, device, args.epochs,
          args.log_interval,
          args.load_weight_path, args.save_graph)
