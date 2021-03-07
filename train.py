import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch._C import device, dtype
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from src.unet_.eval_fun import eval_net
from src.unet_.unet_model import UNet

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, SAT_DIR
from src.preprocessing.esa_compress import compress_esa
from src.preprocessing.load_landsat_esa import return_xy_np_grid, return_x_y_da
from src.unet_.dataset import BasicDataset
#x represents landsat data, and y represents ESA CCI LC.
cfd = {
    "start_year_i": 0,
    "mid_year_i": 19,
    "end_year_i": 24,
    "take_esa_coords": True,
    "use_ffil": True,
    "use_mfd": False,
}

x_da, y_da = return_x_y_da(
    take_esa_coords=cfd["take_esa_coords"],
    use_ffil=cfd["use_ffil"],
    use_mfd=cfd["use_mfd"]
)  # load preprocessed data from netcdfs

x_tr, y_tr = return_xy_np_grid(
    x_da, y_da, year=range(cfd["start_year_i"], cfd["mid_year_i"])
    )  


def train_net(net,
              device,
              data_path,
              epochs=40,
              batch_size=2,
              lr=0.001,
              ):
    
    chern_dataset = BasicDataset(data_path)
    train_loader = DataLoader(dataset=chern_dataset, batch_size=batch_size, shuffle=True)
    
    #Define RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #Define Loss
    criterion = nn.BCEWithLogitsLoss()
    #best_loss
    best_loss = float('inf')
    #Train epochs
    for epoch in range(epochs):
        #Train model
        net.train()
        #Based on batch_size for training
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device =device, dtype=torch.float32)
            #output the prediction result
            pred = net(image)
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            #save the net parameter that make the min loss
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'bets_model.pth')
            #Update the parameter
            loss.backward()
            optimizer.step()

'''
The current problems for implementing UNet are:
1. When do BasicDataset function(in src.unet_.dataset.py), image shpae become 5 dimension (2, 19, 681, 1086, 12) (batch size, yr, y, x, band).
But the dimension in the model is 4 dimension (64, 3, 3, 3);

2. The input size in UNet model should be (N, Cin, H, W): N is a batch size, CC denotes a number of channels, 
H is a height of input planes in pixels, and W is width in pixels (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d). 
But input x and y are (yr, y, x, band), and the order should be changed. 
Maybe permute function in torch.Tensor can help (https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute), but not sure how to use it;

3. Plan to use 3 channels in UNet model, represented RGB images. But the x_tr seems to have 12 bands (3 RGB bands * 4 seasons), and will it conflict with 3 bands in UNet (?);

4. Still working on clarify whether n_classes requal to the number of classified land cover classes that we want.
'''

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=20, bilinear=True)
    net.to(device=device)
    
    data_path = [
            GWS_DATA_DIR / "esa_cci_rois" / f"esa_cci_{year}_chernobyl.geojson"
            for year in range(1992, 2016)
        ]
    train_net(net, device, data_path)
   