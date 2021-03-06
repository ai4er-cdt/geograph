from os.path import splitext
from os import listdir
from glob import glob
import torch
from torch import random
from torch.utils.data import Dataset


import numpy as np
import random
import cv2

from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, SAT_DIR
from src.preprocessing.esa_compress import compress_esa, decompress_esa, FORW_D, REV_D
from src.preprocessing.load_landsat_esa import return_xy_np_grid, y_npa_to_xr, x_npa_to_xr, return_x_y_da


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
    # there are now 24 years to choose from. 
    # train set goes from 0 to 1. # print(x_da.year.values)
    # test_inversibility()
x_tr, y_tr = return_xy_np_grid(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["mid_year_i"])
    )  # load numpy train data.


class BasicDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    #def augment(self, image, flipCode):
        #use cv2.flip to augment data
        #flip = cv2.flip(image, flipCode)
        #return flip

    def __getitem__(self, index):
        image = x_tr
        label = compress_esa(y_tr)
        #image = image.reshape(19, , 512)
        #label = label.reshape(19, 512, 512)
        #Process the label, change the pixel from 255 to 1
        if label.max() > 1:
            label = label / 225
        #Random to do the data augmentation
        #flipCode = random.choice([-1, 0, 1, 2])
        #if flipCode != 2:
            #image = self.augment(image, flipCode)
            #label = self.augment(label, flipCode)
        
        return image, label
    
    def __len__(self):
        #return the training set
        return len(self.data_path)

if __name__ == '__main__':
    train_dataset = BasicDataset([
            GWS_DATA_DIR / "esa_cci_rois" / f"esa_cci_{year}_chernobyl.geojson"
            for year in range(1992, 2016)
        ])
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    batch_size=2,
    shuffle=True)

    for image, label in train_loader:
        print(image.shape)